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

from transformers import BertConfig

def prepare_data(config, tokenizer):
    tmp_train_raw, test_raw = load_ATIS()
    train_raw, dev_raw, test_raw = create_dev_set(tmp_train_raw, test_raw)

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=config['cutoff'])
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    return lang, train_dataset, dev_dataset, test_dataset  # Return tokenizer

def init_model(lang, config):
    config_bert = BertConfig.from_pretrained('bert-base-uncased')  # Or bert-large-uncased
    model = BertForIntentAndSlot.from_pretrained(
        'bert-base-uncased',
        config=config_bert,
        num_intent_labels=len(lang.intent2id),
        num_slot_labels=len(lang.slot2id)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])  # Use AdamW
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    return model, optimizer, criterion_slots, criterion_intents

def train_loop(data_loader, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        sample = sample.to(device) # Move the sample to GPU
        slots, intent = model(sample['utterances'], sample['attention_mask']) #! CHECK THIS
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

def eval_loop(data_loader, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            sample = sample.to(device) # Move the sample to GPU
            
            slots, intents = model(sample['utterances'], sample['attention_mask']) #! CHECK THIS
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
                decoded_token = tokenizer.decode(sample['utterances'][id_seq]).split() # decoding the tokenized input and splitting it
                # utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                gt_slots = gt_slots[1:] # Remove [CLS]
                # utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[1:length].tolist() # first element is [CLS] and last is [SEP]
                
                corrected_tokens = []

                # TRICK seen in this repo: https://github.com/OmarFacchini/NLU-projects/blob/33eea109b1b7f852cec97bb2bfc383ca5e8be753/NLU/part_2/functions.py#L258
                # Adjust tokenization to match original words, replacing extra tokens with 'O' (when there is a ')
                for word in decoded_token:
                    if "'" in word:
                        parts = word.split("'")
                        for part in parts[:-1]:
                            corrected_tokens.append(part)
                            corrected_tokens.append('O')
                        corrected_tokens.append(parts[-1])
                    else:
                        corrected_tokens.append(word)

                # Remove the first token ([CLS]) and the last token ([SEP])
                corrected_tokens = corrected_tokens[1:-1]

                # add padding tokens to ensure the length matches
                while len(corrected_tokens) < len(gt_slots):
                    corrected_tokens.append(lang.slot2id['pad'])
                
                ref_slots.append([(corrected_tokens[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((corrected_tokens[id_el], lang.id2slot[elem]))
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


def get_dataloaders(train_dataset, dev_dataset, test_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size_eval'], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_eval'], collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader


def run_experiments(config, model_class, data_loaders, lang, tokenizer):
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

    # RUN a series of trainings
    for run_idx in tqdm(range(config['runs']), desc="🚀 Runs"):
        
       # 1. Create BertConfig
        try:
            bert_config = BertConfig.from_pretrained("bert-base-uncased")  # Or any other model
        except OSError as e:
            print(f"Error loading model: {e}")
            print(f"Please check if '{config['model_name']}' is a valid Hugging Face model name.")
            continue  # Skip this run if the model can't be loaded
        bert_config.hidden_size = config['bert_hidden_size']
        bert_config.dropout = config['bert_dropout_prob']

        # 2. Instantiate the model
        model = model_class(
            config=bert_config,  # Pass the BertConfig
            num_intent_labels=len(lang.intent2id),
            num_slot_labels=len(lang.slot2id)
        ).to(device)

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

                results_dev, intent_dev, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer)
                losses_dev.append(np.mean(loss_dev))
                f1 = results_dev['total']['f']

                msg = f"📚 Epoch {epoch}/{config['n_epochs']} 🔍 Dev Slot-F1: {f1:.4f}"

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), os.path.join(bin_dir, f'best_{run_idx}.pt'))
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
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang, tokenizer)
        slot_f1s.append(results_test['total']['f'])
        intent_accs.append(intent_test['accuracy'])

        # === Report of the current run ===
        print(f"🧪 Test F1: {results_test['total']['f']:.4f}, Intent Accuracy: {intent_test['accuracy']:.4f}")
        
        # === Store model performance locally
        models_results[run_idx]=results_test['total']['f']
        
        # === Save each run ===
        save_loss_data_per_run(run_idx, sampled_epochs, losses_train, losses_dev,
                               results_test['total']['f'], intent_test['accuracy'], config, exp_dir)

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
