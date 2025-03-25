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

DEVICE = 'cuda'
DEBUG = False

# * HYPERPARAMETERS ------
hid_size = 400 #! MODIFY
emb_size = 300 #! MODIFY

lr = 0.8 # ! MODIFY
clip = 5 # Clip the gradient #? MODIFY
n_epochs = 100
patience_init = 3
# * ------

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")
    
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    
    #print(len(vocab))
    
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    # * DATA LOADING
    
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    # * MODEL SETUP*
    vocab_len = len(lang.word2id)
    
    if DEBUG:
        DEVICE = 'cpu'
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    # * TRAINING

    
    losses_train = []
    losses_dev = []
    ppls_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    patience = patience_init
    
    
    print(f"hidden layers: {hid_size}, emb_size: {emb_size}, lr: {lr}, clip: {clip}, patience: {patience}")
    pbar = tqdm(range(1,n_epochs))
    
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            ppls_dev.append(ppl_dev)
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean
    

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)
    
    model_id = want_to_save_model(best_model) # to choose whether to save the model
    save_training_plot(losses_train, losses_dev, ppls_dev, f"plots/training_plot_{model_id}.png")        
    save_log_csv(model_id, hid_size, emb_size, lr, clip, n_epochs, patience_init, ppl_dev, final_ppl)