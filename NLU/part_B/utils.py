# Add functions or classes used for data loading and preprocessing
import os
import json
from pprint import pprint
import random
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import csv
import matplotlib.pyplot as plt
import os

# ! GLOBAL VARIABLES
device = 'cuda:0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True) # convert words into idices
        self.slot2id = self.lab2id(slots)   # convert labels (fromloc.city_name, toloc.city_name, ecc.) to ids (indices)
        self.intent2id = self.lab2id(intents, pad=False) # intent is just a label so padding is not useful
        self.id2word = {v:k for k, v in self.word2id.items()}   # these 3 functions are the inverse of the previous ones (useful for debugging)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}      # add padding (to fill phrases of different length and make them equally long)
        if unk:
            vocab['unk'] = len(vocab)   # add toke for unknown words (id = leng of vocab -> first available index, NOTE that you are increasing the vocab so every time you get a different id)
        count = Counter(elements)   # get frequency of words
        for k, v in count.items():            # k = word, v = frequency (e.g. "Tony": 3 -> the word "Tony" has 3 occurences)
            if v > cutoff:  # * we consider in the vocabulary just the words having a frequency over the cutoff
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):   # get all the labels (slot and intent)
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)    # also here assign the first available id
        return vocab


class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk', tokenizer=None, max_len=150):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        # We need to tokenize the utterance and map it to the corresponding id
        slots_ids = self.slot_ids[idx]
        intent = self.intent_ids[idx]
        
        words = self.utterances[idx].split()
        slots = []
        
        for w, s in zip(words, slots_ids):
            subwords = self.tokenizer.tokenize(w) # Tokenize the word
            slots.extend([s] * len(subwords)) # Extend the slots list with the same slot id for each subword
        
        tokenized = self.tokenizer(self.utterances[idx], padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        ids = tokenized['input_ids'].squeeze() # Get the input ids
        attention_mask = tokenized['attention_mask'].squeeze() # Get the attention mask
        
        aligned_labels = [lang.slot2id['0']] + [lang.slot2id['0']] + slots # Align the labels with the max length
        
        aligned_labels.extend([lang.slot2id['pad']] * (len(ids) - len(aligned_labels)))
        aligned_labels = aligned_labels[:len(ids)]  # Ensure alignment
            
        sample = {'utterance': ids, 'attention_mask': attention_mask, 'slots': torch.tensor(aligned_labels), 'intent': intent }
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])       # if it is in the mapping, assign the id
                else:
                    tmp_seq.append(mapper[self.unk])    # otherwise set to unknown
            res.append(tmp_seq)
        return res



def load_ATIS():
    def load_data(path):
        dataset = []
        with open(path) as f:
            dataset = json.loads(f.read())
        return dataset

    tmp_train_raw = load_data(os.path.join('../..','ATIS','train.json'))
    test_raw = load_data(os.path.join('../..','ATIS','test.json'))
    # print('Train samples:', len(tmp_train_raw))
    # print('Test samples:', len(test_raw))

    # pprint(tmp_train_raw[0])
    return tmp_train_raw, test_raw
    
def create_dev_set(tmp_train_raw, test_raw):
    portion = 0.10  # use 10% of training set
    
    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []
    
    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    
    return train_raw, dev_raw, test_raw

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


def save_loss_data_per_run(run_idx, run_epochs, run_train_losses, run_dev_losses, 
                           f1, acc, config, exp_dir):
    """
    Salva il file .npz con i punti e il grafico con info della singola run.
    """
    # 1. Salva i dati
    data_path = os.path.join(exp_dir, f"run{run_idx+1}_loss_data.npz")
    np.savez(data_path, epochs=run_epochs, train=run_train_losses, dev=run_dev_losses)

    # 2. Crea il plot
    plt.figure(figsize=(8, 5))
    plt.title(f"Run {run_idx+1} - Train/Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(run_epochs, run_train_losses, label="Train loss")
    plt.plot(run_epochs, run_dev_losses, label="Dev loss")
    plt.legend()

    # 3. Annotazione a lato
    info = (
        f"Hid: {config['hid_size']}, Emb: {config['emb_size']}, LR: {config['lr']}, Clip: {config['clip']}\n"
        f"F1: {round(f1, 3)}, Intent Acc: {round(acc, 3)}"
    )
    plt.text(1.02, 0.5, info, transform=plt.gca().transAxes,
             verticalalignment='center', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.5))

    # 4. Salva il grafico
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, f"run{run_idx+1}_loss_plot.png")
    plt.savefig(plot_path)
    plt.close()
    

def plot_all_runs(all_losses_train, all_losses_dev, all_epochs, exp_dir):
    """
    Crea un grafico con tutte le run, con colori diversi.
    """
    plt.figure(figsize=(10, 6))
    for i, (train, dev, epochs) in enumerate(zip(all_losses_train, all_losses_dev, all_epochs)):
        plt.plot(epochs, train, label=f"Train Run {i+1}")
        plt.plot(epochs, dev, '--', label=f"Dev Run {i+1}")

    plt.title("All Runs - Train/Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    max_loss = max([max(run) for run in all_losses_dev])  # assuming all_losses_dev is a list of list
    plt.ylim(0, max_loss + 0.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "all_runs_plot.png"))
    plt.close()


def log_experiment_summary(timestamp, config, f1s, accs, exp_dir, summary_path="experiments/summary.csv"):
    """
    Logga in un file CSV globale le metriche dell'esperimento.
    """
    # Modifica l'header per includere tutte le nuove configurazioni
    header = ['timestamp', 'hid_size', 'emb_size', 'lr', 'clip', 'runs', 'batch_size_train', 'batch_size_eval',
              'n_epochs', 'patience', 'cutoff', 'bidirectional', 'dropout', 'dropout_rate', 'n_layers',
              'dev_f1_mean', 'dev_f1_std', 'dev_acc_mean', 'dev_acc_std', 'path']
    
    # Prepara la riga da scrivere nel CSV
    row = [
        timestamp, config['hid_size'], config['emb_size'], config['lr'], config['clip'], config['runs'],
        config['batch_size_train'], config['batch_size_eval'], config['n_epochs'], config['patience'], config['cutoff'],
        config['bidirectional'], config['dropout'], config['dropout_rate'], config['n_layers'],
        round(f1s.mean(), 3), round(f1s.std(), 3), round(accs.mean(), 3), round(accs.std(), 3), exp_dir
    ]

    # Controlla se scrivere l'intestazione del file CSV
    write_header = not os.path.exists(summary_path)

    # Scrivi i dati nel file CSV
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)  # Scrivi l'intestazione se il file non esiste
        writer.writerow(row)  # Scrivi la riga con i risultati dell'esperimento
