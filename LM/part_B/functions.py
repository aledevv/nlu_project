import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import math
import numpy as np

DEVICE = 'cuda'


def run_experiment(WEIGHT_TYING, VARIATIONAL_DROPOUT, NMT_AvSGD,
                   train_dataset, dev_dataset, test_dataset, train_batch, hid_size=400, emb_size=300, lr=1.0):

    training_notes = ''
    if WEIGHT_TYING: training_notes += 'Weight Tying,'
    if VARIATIONAL_DROPOUT: training_notes += 'Variational Dropout,'
    if NMT_AvSGD: training_notes += 'Non-monotonically Triggered AvSGD,'
    if training_notes == '': training_notes = 'Vanilla'

    eval_batch = train_batch * 2
    train_loader = DataLoader(train_dataset, batch_size=train_batch, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=eval_batch, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    vocab_len = len(lang.word2id)
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], use_weight_tying=WEIGHT_TYING).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    losses_train, losses_dev, ppls_dev, sampled_epochs = [], [], [], []
    best_ppl = math.inf
    best_model = None
    patience = patience_init

    print(f"\n[Run] Reg: {training_notes} | batch: {train_batch} | lr: {lr} | hid: {hid_size} | emb: {emb_size}")

    for epoch in tqdm(range(1, n_epochs), desc="Training"):
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss).mean())
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        losses_dev.append(np.asarray(loss_dev).mean())
        ppls_dev.append(ppl_dev)

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = patience_init
        else:
            patience -= 1

        if patience <= 0:
            break

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl:', final_ppl)

    model_id = want_to_save_model(best_model)
    save_training_plot(losses_train, losses_dev, ppls_dev, f"plots/training_plot_{model_id}.png")
    save_log_csv(model_id, hid_size, emb_size, lr, clip, n_epochs, patience_init, ppl_dev, final_ppl, training_notes)


def train_loop(data, optimizer, criterion, model, clip=5):
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