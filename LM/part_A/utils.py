# Add functions or classes used for data loading and preprocessing
import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import curses
import os
import re
from functions import DEVICE

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

# * Data loading functions
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


def collate_fn(data, pad_token):
    def merge(sequences):
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
    return os.path.join(directory, f"model_{next_number}.pt")
        
        
def want_to_save_model(model):
    choice = curses.wrapper(menu)  # Run the menu inside curses wrapper
    
    if "Yes" in choice:
        save_path = get_next_model_name(save_dir)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at: {save_path}")
    else:
        print("Model not saved.")
    