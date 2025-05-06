# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
import torch.optim as optim
from functions import *
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import curses
import os
import re
import itertools
from model import LM_LSTM
from torch.utils.data import DataLoader



DEVICE = 'cuda'
DEBUG = False

# * HYPERPARAMETERS ------
hid_size = 400 #! MODIFY # default (400)
emb_size = 300 #! MODIFY # default (300)

lr = 1 #! MODIFY
clip = 5 # Clip the gradient
n_epochs = 100
patience_init = 3
train_batch = 64 #? (64)

#* regularizarion techniques to use
WEIGHT_TYING = True                 
VARIATIONAL_DROPOUT = True
NMT_AvSGD = False

training_notes = '(first 2 techniques)'  #TODO Notes that will be reported in the csv
# * ------

# EXPERIMENTS

# -------------------- ESPERIMENTI COMPLETI --------------------

configs = list(itertools.product(
    [False, True],  # WEIGHT_TYING
    [False, True],  # VARIATIONAL_DROPOUT
    [False, True]   # NMT_AvSGD
))

batch_sizes = [32, 64, 128]
learning_rates = [1.0, 0.5, 0.1]



if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    train_raw = read_file("../dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("../dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("../dataset/PennTreeBank/ptb.test.txt")
    
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    
    #print(len(vocab))
    
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    # * DATA LOADING
    
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    # * MODEL SETUP*
    vocab_len = len(lang.word2id)
    
    if DEBUG:
        DEVICE = 'cpu'
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], use_weight_tying=WEIGHT_TYING).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    # * TRAINING
    
    # First stage: all combinations regularization + batch + lr
    for weight_tying, var_dropout, nmt_avg in configs:
        for bsz in batch_sizes:
            for lr_val in learning_rates:
                run_experiment(weight_tying, var_dropout, nmt_avg, train_dataset, dev_dataset, test_dataset, train_batch=bsz, lr=lr_val)
       
    # Second stage: all regularizations + different combinations of hidden and embedding sizes         
    final_configs = [(512, 300), (400, 400), (300, 200), (600, 600)]
    for hid, emb in final_configs:
        run_experiment(True, True, True, train_dataset, dev_dataset, test_dataset, train_batch=64, hid_size=hid, emb_size=emb, lr=1.0)