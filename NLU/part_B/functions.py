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
from datetime import datetime
from transformers import BertTokenizerFast

def prepare_data(config):
    tmp_train_raw, test_raw = load_ATIS()
    train_raw, dev_raw, test_raw = create_dev_set(tmp_train_raw, test_raw)

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=config['cutoff'])

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    train_dataset = BertAtisDataset(train_raw, tokenizer, lang)
    dev_dataset = BertAtisDataset(dev_raw, tokenizer, lang)
    test_dataset = BertAtisDataset(test_raw, tokenizer, lang)

    return lang, train_dataset, dev_dataset, test_dataset


def get_dataloaders(train_dataset, dev_dataset, test_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size_train'], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_eval'], collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader


def run_experiments(config, model_class, data_loaders, lang):
    # === Init experiment folder ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_dir = os.path.join("experiments", f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    bin_dir = os.path.join(exp_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    train_loader, dev_loader, test_loader = data_loaders
    all_losses_train, all_losses_dev, all_epochs = [], [], []
    slot_f1s, intent_accs = [], []

    for run_idx in tqdm(range(config['runs']), desc="🚀 Runs"):

        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion_slots = torch.nn.CrossEntropyLoss(ignore_index=lang.slot2id['pad'])
        criterion_intents = torch.nn.CrossEntropyLoss()

        best_f1 = 0
        patience = config['patience']
        models_results = {}
        losses_train, losses_dev, sampled_epochs = [], [], []

        # START TRAINING for the current run
        print(f"🌀 Starting run {run_idx+1}/{config['runs']}")
        for epoch in range(1, config['n_epochs'] + 1):
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)

            if epoch % 2 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(sum(loss)/len(loss))

                results_dev, intent_dev, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                losses_dev.append(sum(loss_dev)/len(loss_dev))


                f1 = results_dev['total']['f']
                acc = intent_dev['accuracy']
                msg = f"📚 Epoch {epoch}/{config['n_epochs']} 🔍 Dev Slot-F1: {f1:.4f} Accuracy: {acc:.4f}"

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), os.path.join(bin_dir, f'best_{run_idx}.pt')) # ? if model is better
                    patience = config['patience']
                    msg += "✅ New best F1! Resetting patience."
                else:
                    patience -= 1
                    msg += f"⏳ No improvement. Patience left: {patience}"

                print(msg)
                
                if patience <= 0:
                    print("🛑 Early stopping")
                    break

        # === Evaluation ===
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        slot_f1s.append(results_test['total']['f'])
        intent_accs.append(intent_test['accuracy'])
        
        # === Report of the current run ===
        print(f"🧪 Test F1: {results_test['total']['f']:.4f}, Intent Accuracy: {intent_test['accuracy']:.4f}")
        
        # === Store model performance locally
        models_results[run_idx] = {}
        models_results[run_idx]["score"]=results_test['total']['f']+intent_test['accuracy']
        models_results[run_idx]["intent_acc"]=intent_test['accuracy']
        models_results[run_idx]["slot_f1"]=results_test['total']['f']
        
        # === Save each run ===
        save_loss_data_per_run(run_idx, sampled_epochs, losses_train, losses_dev,
                               results_test['total']['f'], intent_test['accuracy'], config, exp_dir)

        all_losses_train.append(losses_train)
        all_losses_dev.append(losses_dev)
        all_epochs.append(sampled_epochs)

    
    # === KEEP BEST MODEL ===
    best_run_idx = max(models_results, key=lambda i: models_results[i]["score"])

    # Save the best template as best.pt
    best_model_path = os.path.join(bin_dir, f'best_{best_run_idx}.pt')
    best_model_save_path = os.path.join(exp_dir, 'best.pt')
    shutil.copy(best_model_path, best_model_save_path) # Copy the best model to the experiment root folder as 'best.pt'
    shutil.rmtree(os.path.join(exp_dir, 'bin')) # Delete the folder 'bin/' with the weights of the runs
    
    # === Summary plot and log ===
    plot_all_runs(all_losses_train, all_losses_dev, all_epochs, exp_dir)
    log_experiment_summary(timestamp, config, models_results[best_run_idx]["slot_f1"], models_results[best_run_idx]["intent_acc"], exp_dir) # save the summary of the best run

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
        optimizer.zero_grad()
        slot_logits, intent_logits = model(
            input_ids=sample['input_ids'],
            attention_mask=sample['attention_mask'],
            token_type_ids=sample['token_type_ids']
        )

        # Compute intent loss
        loss_intent = criterion_intents(intent_logits, sample['intent_labels'])

        # Compute slot loss (only for real tokens, not subtokens/padding)
        active_loss = sample['slot_label_mask'].view(-1)
        active_logits = slot_logits.view(-1, slot_logits.shape[-1])[active_loss]
        active_labels = sample['slot_labels'].view(-1)[active_loss]
        loss_slot = criterion_slots(active_logits, active_labels)

        loss = loss_intent + loss_slot
        loss_array.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data:
            slot_logits, intent_logits = model(
                input_ids=sample['input_ids'],
                attention_mask=sample['attention_mask'],
                token_type_ids=sample['token_type_ids']
            )

            loss_intent = criterion_intents(intent_logits, sample['intent_labels'])
            active_loss = sample['slot_label_mask'].view(-1)
            active_logits = slot_logits.view(-1, slot_logits.shape[-1])[active_loss]
            active_labels = sample['slot_labels'].view(-1)[active_loss]
            loss_slot = criterion_slots(active_logits, active_labels)

            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            # Intent prediction
            pred_intents = torch.argmax(intent_logits, dim=1).tolist()
            ref_intents.extend([lang.id2intent[i.item()] for i in sample['intent_labels']])
            hyp_intents.extend([lang.id2intent[i] for i in pred_intents])

            # Slot prediction
            slot_preds = torch.argmax(slot_logits, dim=2).cpu().tolist()
            slot_labels = sample['slot_labels'].cpu().tolist()
            masks = sample['slot_label_mask'].cpu().tolist()
            input_ids = sample['input_ids'].cpu().tolist()

            for pred, label, mask, ids in zip(slot_preds, slot_labels, masks, input_ids):
                tokens = lang.id2word
                words = [tokens.get(i, 'unk') for i, m in zip(ids, mask) if m == 1]
                gt = [lang.id2slot[i] for i, m in zip(label, mask) if m == 1]
                pr = [lang.id2slot[i] for i, m in zip(pred, mask) if m == 1]
                ref_slots.append(list(zip(words, gt)))
                hyp_slots.append(list(zip(words, pr)))

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array
