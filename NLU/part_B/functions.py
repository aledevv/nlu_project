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

from transformers import BertTokenizerFast, BertConfig, BertModel

def prepare_data(config):
    tmp_train_raw, test_raw = load_ATIS()
    train_raw, dev_raw, test_raw = create_dev_set(tmp_train_raw, test_raw)

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=config['cutoff'])
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Or bert-large-uncased
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    return lang, train_dataset, dev_dataset, test_dataset, tokenizer  # Return tokenizer

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
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        intent_labels = batch['intent'].to(device)
        slot_labels = batch['slot_ids'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, intent_labels=intent_labels, slot_labels=slot_labels)
        loss = outputs[0]
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return total_loss / len(data_loader)

def eval_loop(data_loader, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    total_loss = 0
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent'].to(device)
            slot_labels = batch['slot_ids'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, intent_labels=intent_labels, slot_labels=slot_labels)
            loss = outputs[0]
            total_loss += loss.item()

            intent_logits, slot_logits = outputs[1], outputs[2]

            # Intent Evaluation
            intent_preds = torch.argmax(intent_logits, axis=1).cpu().numpy()
            intent_labels = intent_labels.cpu().numpy()
            ref_intents.extend([lang.id2intent[i] for i in intent_labels])
            hyp_intents.extend([lang.id2intent[i] for i in intent_preds])

            # Slot Evaluation (Handle Sub-word Tokenization!)
            slot_preds = torch.argmax(slot_logits, axis=2).cpu().numpy()
            slot_labels = slot_labels.cpu().numpy()
            
            input_ids_np = input_ids.cpu().numpy()

            for i in range(len(slot_preds)):  # Loop through each sequence in the batch
                
                # Decode the input_ids to get tokens
                tokens = tokenizer.convert_ids_to_tokens(input_ids_np[i], skip_special_tokens=True)
                
                #  Align predictions and labels, considering sub-word tokenization
                aligned_predictions = []
                aligned_labels = []
                
                current_word_preds = []
                current_word_labels = []

                for j, token in enumerate(tokens):
                     if token.startswith("##"):  # Part of a sub-word
                         current_word_preds.append(lang.id2slot.get(slot_preds[i][j+1], 'O')) # +1 because of CLS token
                         current_word_labels.append(lang.id2slot.get(slot_labels[i][j+1], 'O'))
                     else:  # Start of a new word
                         if current_word_preds:  # Process the previous word
                             aligned_predictions.append(max(set(current_word_preds), key=current_word_preds.count))  # Or take the first, or any other strategy
                             aligned_labels.append(max(set(current_word_labels), key=current_word_labels.count))
                         current_word_preds = [lang.id2slot.get(slot_preds[i][j+1], 'O')]
                         current_word_labels = [lang.id2slot.get(slot_labels[i][j+1], 'O')]
                if current_word_preds:  # Process the last word
                     aligned_predictions.append(max(set(current_word_preds), key=current_word_preds.count))
                     aligned_labels.append(max(set(current_word_labels), key=current_word_labels.count))

                # Get original words
                original_words = utterance[i].split() # TODO FIXME

                # Ensure lengths match (handle potential tokenizer edge cases)
                min_len = min(len(original_words), len(aligned_predictions), len(aligned_labels))
                
                ref_slots.append([(original_words[k], aligned_labels[k]) for k in range(min_len)])
                hyp_slots.append([(original_words[k], aligned_predictions[k]) for k in range(min_len)])
                
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, total_loss / len(data_loader)


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
