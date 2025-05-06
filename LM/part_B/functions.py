import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from model import LM_LSTM
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from utils import *
DEVICE = 'cuda'

def run_experiment(config, weight_tying, variational_dropout, train_dataset, nt_avsgd, dev_dataset, test_dataset, lang,):
    patience_init = config['patience_init']
    clip = config['clip']
    n_epochs = config['n_epochs']
    training_notes = config['training_notes']
    train_batch = config['train_batch']
    lr = config['lr']
    emb_size = config['emb_size']
    hid_size = config['hid_size']
    
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    # * MODEL SETUP*
    vocab_len = len(lang.word2id)
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], use_weight_tying=weight_tying, use_variational_dropout=variational_dropout).to(DEVICE)
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
    
    if weight_tying:
        training_notes = training_notes + ' Weight Tying,'
    if variational_dropout:
        training_notes = training_notes + ' Variational Dropout,'
    if nt_avsgd:
        training_notes = training_notes + ' Non-monotonically Triggered AvSGD,'
        print("Using Non-monotonically Triggered AvSGD (NT-AvSGD)")
    
    print(f"hidden layers: {hid_size}, emb_size: {emb_size}, lr: {lr}, clip: {clip}, patience: {patience}, batch_size: {train_batch}, notes: {training_notes if training_notes != '' else 'None'}")
    pbar = tqdm(range(1,n_epochs))
    
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, dev_loader, optimizer, criterion_train, criterion_eval, model, clip, use_nt_avsgd=nt_avsgd)    
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
    save_log_csv(model_id, hid_size, emb_size, lr, clip, n_epochs, patience_init, ppl_dev, final_ppl, training_notes)
    return final_ppl

def train_loop(data, eval_data, optimizer, criterion, criterion_eval, model, clip=5, use_nt_avsgd=False, logging_interval=1, non_monotone_threshold=5):
    model.train()
    loss_array = []
    ppls_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
        
        if use_nt_avsgd:
            running_sum = None  # Accumulates the parameter vectors
            running_count = 0   # Counts the number of updates accumulated after model. T
            
            if (model.k % logging_interval == 0) and (model.k > 0):
                if eval_data is None:
                    raise ValueError("Provide eval data if NT_AvSGD is True")
                v, loss_dev = eval_loop(eval_data, criterion_eval, model)
                model.train()
                if (v > non_monotone_threshold) and (model.logs and v > model.logs[-1]):
                    model.T = model.k
                model.logs.append(v)
                model.t += 1
            model.k += 1
            
            
            if model.k > model.T:   #? Accumulate weights after T iteration
                with torch.no_grad():
                    params_vector = torch.cat([p.detach().view(-1) for p in model.parameters()])
                    if running_sum is None:
                        running_sum = params_vector.clone()
                    else:
                        running_sum += params_vector
                running_count += 1

    #* https://github.com/jo-valer/ASGD-optimizer/blob/main/ntasgd.py
    # If NT-AvSGD is active, calculate the average of accumulated weights and apply it to the model
    if use_nt_avsgd and running_sum is not None and running_count > 0:
        w_avg = running_sum / running_count
        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                num_params = p.numel()
                new_val = w_avg[offset:offset + num_params].view_as(p)
                p.copy_(new_val)
                offset += num_params            

    return sum(loss_array)/sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)