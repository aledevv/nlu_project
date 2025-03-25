import copy
import curses
import os
import re
import csv
from tabulate import tabulate

DEVICE = 'cuda'

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids
def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

import torch
import torch.utils.data as data

class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res


from functools import partial
from torch.utils.data import DataLoader

def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item

# * TO SAVE THE MODEL AFTER THE EVALUATION

save_dir = "model_bin"

# Fancy selection menu
def menu(stdscr):
    curses.curs_set(0)  # Hide cursor
    options = ["Yes (Save model)", "No"]
    current_idx = 0

    while True:
        stdscr.clear()

        stdscr.addstr(0, 0, "Do you like the model?", curses.A_BOLD)
        for idx, option in enumerate(options):
            if idx == current_idx:
                stdscr.addstr(idx + 2, 2, f"> {option}", curses.A_REVERSE)  # Highlight selection
            else:
                stdscr.addstr(idx + 2, 2, f"  {option}")

        key = stdscr.getch()

        if key == curses.KEY_UP and current_idx > 0:
            current_idx -= 1
        elif key == curses.KEY_DOWN and current_idx < len(options) - 1:
            current_idx += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:  # Enter key
            return options[current_idx]


def get_next_model_name(directory):
    """Finds the next available model filename in the directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create folder if it doesn't exist

    existing_files = [f for f in os.listdir(directory) if re.match(r"model_\d+\.pt", f)]
    existing_numbers = [int(re.search(r"model_(\d+)\.pt", f).group(1)) for f in existing_files]

    next_number = max(existing_numbers, default=0) + 1  # Increment highest number
    return os.path.join(directory, f"model_{next_number}.pt"), next_number
        
        
def want_to_save_model(model):
    #choice = curses.wrapper(menu)  # Run the menu inside curses wrapper
    choice = ['Yes']
    
    if "Yes" in choice:
        save_path, model_id = get_next_model_name(save_dir)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at: {save_path}")
    else:
        print("Model not saved.")
    
    return model_id
        

# * TO STORE LOG (configuration and performance of each training)


def save_log_txt(model_id=0, hidden_size=None, emb_size=None, learning_rate=None, clip=None, epochs=0, patience=0, dev_ppl=0, final_ppl=0, log_file='training_log.txt'):
    # Definisce i nomi delle colonne
    fieldnames = ['Model ID', 'Hidden Size', 'Embedding Size', 'Learning Rate', 
                  'Gradient Clip', 'Epochs', 'Patience', 'Dev PPL', 'Final PPL']

    # Dati da salvare
    log_entry = [model_id, hidden_size, emb_size, learning_rate, clip, epochs, patience, dev_ppl, final_ppl]

    # Controlla se il file esiste già
    file_exists = os.path.exists(log_file)

    # Se il file non esiste, crea l'intestazione
    with open(log_file, mode='a') as f:
        if not file_exists:
            f.write(tabulate([fieldnames], tablefmt="grid") + "\n")

        # Scrive i dati nella tabella
        f.write(tabulate([log_entry], tablefmt="grid") + "\n")

    # Stampa il log a schermo
    print("\nLog salvato con successo! 📊")
    print(tabulate([log_entry], headers=fieldnames, tablefmt="grid"))


# * FUNCTION TO PLOT TRAINING
import matplotlib.pyplot as plt

def save_training_plot(losses_train, losses_dev, ppls_dev, filename="training_plot.png"):
    """
    Saves the plot showing the evolution of the loss and perplexity during training and eval
    """
    epochs = range(1, len(losses_train) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot della Training Loss
    ax[0].plot(epochs, losses_train, marker='o', linestyle='-', label="Training Loss", color='b')
    ax[0].plot(epochs, losses_dev, marker='x', linestyle='--', label="Validation Loss", color='r')
    ax[0].set_title("Loss during Training")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    # Plot della Perplexity di Validazione
    ax[1].plot(epochs, ppls_dev, marker='o', linestyle='-', label="Validation PPL", color='g')
    ax[1].set_title("Perplexity on Validation Set")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Perplexity")
    ax[1].legend()
    ax[1].grid(True)

    # Salva il plot nel file specificato
    plt.savefig(filename)
    plt.close()  # Chiude la figura per liberare memoria
    print(f"Plot salvato come {filename}")
