import itertools
import torch.optim as optim
from functions import *
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
from model import LM_LSTM

DEBUG = False

USE_ADAM = False
USE_DROPOUT = False

# Grid di valori da testare
hid_sizes = [400]
emb_sizes = [300]
lrs = [0.5, 1.0, 2]
lrs_adam = [0.00001, 0.0001, 0.001, 0.01]
clips = [5]
batch_sizes = [16, 32, 64]

n_epochs = 100
patience_init = 3

log_file_name = "training_log_vanilla.csv"

# Dati
train_raw = read_file("../dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("../dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("../dataset/PennTreeBank/ptb.test.txt")
vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
lang = Lang(train_raw, ["<pad>", "<eos>"])
vocab_len = len(lang.word2id)

# Ricerca combinazioni
best_ppl = float('inf')
best_config = None

i=0
total_combinations = len(hid_sizes) * len(emb_sizes) * len(lrs) * len(clips) * len(batch_sizes)

for hid_size, emb_size, lr, clip, training_batch_size in itertools.product(hid_sizes, emb_sizes, lrs, clips, batch_sizes):
    print(f"\n🔍 Testing config {i}/{total_combinations}: hid={hid_size}, emb={emb_size}, lr={lr}, clip={clip}, batch={training_batch_size}")
    
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    train_loader = DataLoader(train_dataset, batch_size=training_batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], use_dropout=USE_DROPOUT)
    model.to('cuda')
    
    if USE_ADAM:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    losses_train, losses_dev, ppls_dev, sampled_epochs = [], [], [], []
    best_model = None
    patience = patience_init
    local_best_ppl = float('inf')

    for epoch in tqdm(range(1, n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            ppls_dev.append(ppl_dev)
            if ppl_dev < local_best_ppl:
                local_best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cuda')
                patience = patience_init
            else:
                patience -= 1
            if patience <= 0:
                break

    # Test finale
    best_model.to('cuda')
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print(f"✅ Final Test PPL: {final_ppl:.3f}")

    # Salvataggio
    model_id = want_to_save_model(best_model)
    save_training_plot(losses_train, losses_dev, ppls_dev, f"plots/training_plot_{model_id}.png")        
    save_log_csv(model_id, hid_size, emb_size, lr, clip, n_epochs, patience_init, local_best_ppl, final_ppl, log_file=log_file_name)

    # Aggiorna migliore config
    if final_ppl < best_ppl:
        best_ppl = final_ppl
        best_config = (hid_size, emb_size, lr, clip, training_batch_size)

print("\n🏆 --- BEST CONFIGURATION FOUND ---")
print(f"Hidden size: {best_config[0]}, Emb size: {best_config[1]}, LR: {best_config[2]}, Clip: {best_config[3]}, Batch size: {best_config[4]}")
print(f"Best Final Test Perplexity: {best_ppl:.3f}")
