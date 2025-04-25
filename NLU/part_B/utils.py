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
from transformers import BertTokenizer, BertModel

# ! GLOBAL VARIABLES
device = 'cuda:0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0


from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class BERTJointDataset(Dataset):
    """
    PyTorch Dataset for joint Intent Detection and Slot Filling with BERT.
    Expects raw examples of the form:
      { 'utterance': "word1 word2 ...", 'slots': ["O", "B-LOC", ...], 'intent': "intent_label" }
    """

    def __init__(
        self,
        raw_data,
        tokenizer: BertTokenizerFast,
        intent2id: dict,
        slot2id: dict,
        max_len: int = 64,
    ):
        self.tokenizer = tokenizer
        self.intent2id = intent2id
        self.slot2id = slot2id
        self.max_len = max_len
        self.pad_label_id = slot2id.get('PAD', -100)

        self.features = []
        for ex in raw_data:
            words = ex['utterance'].split()
            slot_labels = ex['slots']
            intent_label = ex['intent']

            # Tokenize and align slots
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors=None,
            )
            word_ids = encoding.word_ids()

            # Align slot labels to sub-tokens
            aligned_slots = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_slots.append(self.pad_label_id)
                else:
                    label = slot_labels[word_idx]
                    if word_idx != prev_word_idx:
                        aligned_slots.append(self.slot2id[label])
                    else:
                        # Inside a word: make sure 'B-' becomes 'I-'
                        if label.startswith('B-'):
                            label = 'I-' + label[2:]
                        aligned_slots.append(self.slot2id.get(label, self.pad_label_id))
                    prev_word_idx = word_idx

            # Intent ID
            intent_id = self.intent2id[intent_label]

            feature = {
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'token_type_ids': torch.tensor(encoding['token_type_ids'], dtype=torch.long),
                'slot_labels': torch.tensor(aligned_slots, dtype=torch.long),
                'intent_label': torch.tensor(intent_id, dtype=torch.long),
            }
            self.features.append(feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def get_label_maps(train_data):
    """
    Build intent2id and slot2id mappings from the training set.
    Adds a special 'PAD' label for slot padding (id = -100 by default).

    Args:
        train_data: list of dicts with keys 'intent' and 'slots'

    Returns:
        intent2id: dict mapping intent label to integer
        slot2id: dict mapping slot label to integer (+ 'PAD')
    """
    intents = sorted({ex['intent'] for ex in train_data})
    intent2id = {intent: idx for idx, intent in enumerate(intents)}

    slots = sorted({slot for ex in train_data for slot in ex['slots']})
    slot2id = {label: idx for idx, label in enumerate(slots)}
    slot2id['PAD'] = -100

    return intent2id, slot2id


def load_tokenizer(model):
    return BertTokenizerFast.from_pretrained(model) # Download the tokenizer


def load_ATIS():
    def load_data(path):
        dataset = []
        with open(path) as f:
            dataset = json.loads(f.read())
        return dataset

    tmp_train_raw = load_data(os.path.join('../..','ATIS','train.json'))
    test_raw = load_data(os.path.join('../..','ATIS','test.json'))

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
