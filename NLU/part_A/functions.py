# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from utils import *
import torch.nn as nn
from conll import evaluate
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import *
from tqdm import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt
import os
import shutil

def prepare_data(config):
    tmp_train_raw, test_raw = load_ATIS()
    train_raw, dev_raw, test_raw = create_dev_set(tmp_train_raw, test_raw)

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=config['cutoff'])

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    return lang, train_dataset, dev_dataset, test_dataset


def get_dataloaders(train_dataset, dev_dataset, test_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size_eval'], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_eval'], collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader


def init_model(lang, config):
    model = ModelIAS(
        config['hid_size'],
        out_slot=len(lang.slot2id),
        out_int=len(lang.intent2id),
        emb_size=config['emb_size'],
        vocab_len=len(lang.word2id),
        pad_index=PAD_TOKEN
    ).to(device)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    return model, optimizer, criterion_slots, criterion_intents


def run_experiments(config, model_class, data_loaders, lang):
    # === Init experiment folder ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    flags = f"bi{config['bidirectional']}_do{config['dropout']}"
    exp_dir = os.path.join("experiments", f"exp_{timestamp}_{flags}")
    os.makedirs(exp_dir, exist_ok=True)
    bin_dir = os.path.join(exp_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # === Unpack data ===
    train_loader, dev_loader, test_loader = data_loaders
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    all_losses_train, all_losses_dev, all_epochs = [], [], []
    slot_f1s, intent_accs = [], []

    # RUN a series of trainings
    for run_idx in tqdm(range(config['runs']), desc="🚀 Runs"):
        
        # Instantiating the model
        model = model_class(
            config['hid_size'], out_slot, out_int, config['emb_size'],
            vocab_len, config['n_layers'], config['bidirectional'], config['dropout'],
            config['dropout_rate'], pad_index=PAD_TOKEN,
        ).to(device)
        model.apply(init_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion_slots = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = torch.nn.CrossEntropyLoss()

        patience = config['patience']
        models_results={}
        best_f1 = 0
        losses_train, losses_dev, sampled_epochs = [], [], []

        # START TRAINING for the current run
        print(f"🌀 Starting run {run_idx+1}/{config['runs']}")
        for epoch in range(1, config['n_epochs'] + 1):
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)

            if epoch % 5 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.mean(loss))

                results_dev, intent_dev, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                losses_dev.append(np.mean(loss_dev))
                f1 = results_dev['total']['f']

                msg = f"📚 Epoch {epoch}/{config['n_epochs']} 🔍 Dev Slot-F1: {f1:.4f}"

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), os.path.join(bin_dir, f'best_{run_idx}.pt')) # ? if model is 
                    patience = config['patience']
                    msg += " ✅ New best F1! Resetting patience."
                else:
                    patience -= 1
                    msg += f" ⚠️ No improvement. Patience left: {patience}"

                print(msg)

                if patience <= 0:
                    print("🛑 Early stopping triggered!")
                    break


        # === Evaluation ===
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        slot_f1s.append(results_test['total']['f'])
        intent_accs.append(intent_test['accuracy'])

        # === Report of the current run ===
        print(f"🧪 Test F1: {results_test['total']['f']:.4f}, Intent Accuracy: {intent_test['accuracy']:.4f}")
        
        # === Store model performance locally
        models_results[run_idx]=results_test['total']['f']
        
        # === Save each run ===
        save_loss_data_per_run(run_idx, sampled_epochs, losses_train, losses_dev,
                               best_f1, intent_test['accuracy'], config, exp_dir)

        all_losses_train.append(losses_train)
        all_losses_dev.append(losses_dev)
        all_epochs.append(sampled_epochs)

    # === Summary plot and log ===
    plot_all_runs(all_losses_train, all_losses_dev, all_epochs, exp_dir)
    log_experiment_summary(timestamp, config, np.array(slot_f1s), np.array(intent_accs), exp_dir)
    
    # === KEEP BEST MODEL ===
    best_run_idx = max(models_results, key=models_results.get)

    # Save the best template as best.pt
    best_model_path = os.path.join(bin_dir, f'best_{best_run_idx}.pt')
    best_model_save_path = os.path.join(exp_dir, 'best.pt')
    shutil.copy(best_model_path, best_model_save_path) # Copy the best model to the experiment root folder as 'best.pt'
    shutil.rmtree(os.path.join(exp_dir, 'bin')) # Delete the folder 'bin/' with the weights of the runs

    return slot_f1s, intent_accs, all_losses_train, all_losses_dev, all_epochs


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
                    
                    

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array
