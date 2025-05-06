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
from model import LM_LSTM
from itertools import product

DEVICE = 'cuda'

# * HYPERPARAMETERS ------
base_config = {
    'patience_init': 3,
    'clip': 5,
    'n_epochs': 100,
    'training_notes': 'logging interval = 1 and non monotonic=5',
    'train_batch': 32,
    'lr': 1.0,
    'emb_size': 400,
    'hid_size': 400
}

#* regularizarion techniques to use
WEIGHT_TYING = True                 
VARIATIONAL_DROPOUT = True
NMT_AvSGD = True


# * KIND OF EXPERIMENTS ------
TEST_LR = True
TEST_SIZE = False
TEST_BATCH = False

# * ------

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
    
    
    if TEST_LR:
        # * TRAINING
        bools = [False, True]
        tech_combinations = list(product(bools, repeat=3))  # (WT, VD, NT)

        # list of learning rates to test
        learning_rates = [5.0, 3.0, 2.0, 1.0, 0.5, 0.1]

        # * Loop through all combinations of techniques vaerying the learning rate
        for wt, vd, nt in tech_combinations:
            name = "vanilla" if not any([wt, vd, nt]) else f"WT={wt}_VD={vd}_NT={nt}"
            base_config["training_notes"] = name
            
            for lr in learning_rates:
                config = base_config.copy()
                config['lr'] = lr
                print(f"\n🚀 Running experiment: {name} | lr={lr}")
                run_experiment(config=config,
                            weight_tying=wt,
                            variational_dropout=vd,
                            nt_avsgd=nt,
                            train_dataset=train_dataset,
                            dev_dataset=dev_dataset,
                            test_dataset=test_dataset,
                            lang=lang) 
    
    if TEST_SIZE:
        # * For the best setting now we change embedding size and hidden size
        model_sizes = [(400, 300), (600, 600), (512, 300), (300, 512)]
        
        for emb_size, hid_size in model_sizes:
            config = base_config.copy()
            config['emb_size'] = emb_size
            config['hid_size'] = hid_size
            config['lr']=0.1 #! TO BE SET
            print(f"\n🚀 Running experiment: {name} | emb_size={emb_size} | hid_size={hid_size}")
            run_experiment(config=config,
                        weight_tying=True,
                        variational_dropout=True,
                            nt_avsgd=True,
                            train_dataset=train_dataset,
                            dev_dataset=dev_dataset,
                            test_dataset=test_dataset,
                            lang=lang)
            
    if TEST_BATCH:
        # * For the best setting now we change the batch size
        batch_sizes = [16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            config = base_config.copy()
            config['train_batch'] = batch_size
            config['lr']=0.1 # ! SET
            config['emb_size'] = emb_size # ! SET
            config['hid_size'] = hid_size # ! SET
            print(f"\n🚀 Running experiment: {name} | batch_size={batch_size}")
            run_experiment(config=config,
                        weight_tying=True,
                        variational_dropout=True,
                            nt_avsgd=True,
                            train_dataset=train_dataset,
                            dev_dataset=dev_dataset,
                            test_dataset=test_dataset,
                            lang=lang)