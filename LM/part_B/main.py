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

DEVICE = 'cuda'

# * HYPERPARAMETERS ------
base_config = {
    'patience_init': 3,
    'clip': 5,
    'n_epochs': 100,
    'training_notes': '',
    'train_batch': 64,
    'lr': 1.0,
    'emb_size': 400,
    'hid_size': 400
}

# * KIND OF EXPERIMENTS ------
TEST_LR = True
TEST_SIZE = True
TEST_BATCH = True

if __name__ == "__main__":
    # Load datasets
    train_raw = read_file("../dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("../dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("../dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Load experiment state
    state = load_state()
    completed = set(tuple(e) for e in state['completed'])
    best_ppl = state['best_ppl']
    best_config = state['best_config']

    if TEST_LR:
        learning_rates = [3.0, 2.0, 1.0, 0.5]
        technique_progression = [
            (False, False, False),        # Vanilla
            (True, False, False),         # + Weight Tying
            (True, True, False),          # + Variational Dropout
            (True, True, True)            # + NMT_AvSGD
        ]
        total_combinations = len(technique_progression) * len(learning_rates)
        experiment_count = 0

        for wt, vd, nt in technique_progression:
            name = "vanilla"
            if wt and not vd and not nt:
                name = "WT"
            elif wt and vd and not nt:
                name = "WT+VD"
            elif wt and vd and nt:
                name = "WT+VD+NT"

            for lr in learning_rates:
                step_id = ('TEST_LR', name, lr)
                if step_id in completed:
                    print(f"⏩ Skipping {step_id}, already completed.")
                    continue

                experiment_count += 1
                config = base_config.copy()
                config["lr"] = lr
                config["training_notes"] = name
                print(f"\n🚀 [TEST_LR] Running experiment: {name} | lr={lr} - {experiment_count}/{total_combinations}")
                final_ppl = run_experiment(
                    config=config,
                    weight_tying=wt,
                    variational_dropout=vd,
                    nt_avsgd=nt,
                    train_dataset=train_dataset,
                    dev_dataset=dev_dataset,
                    test_dataset=test_dataset,
                    lang=lang
                )

                if final_ppl < best_ppl:
                    best_ppl = final_ppl
                    best_config = config.copy()
                    best_config.update({'weight_tying': wt, 'variational_dropout': vd, 'nt_avsgd': nt})

                state['completed'].append(step_id)
                state['best_config'] = best_config
                state['best_ppl'] = best_ppl
                save_state(state)

    if TEST_SIZE:
        model_sizes = [(400, 300), (600, 600), (512, 300), (300, 512)]
        total_combinations = len(model_sizes)
        i = 0

        for emb_size, hid_size in model_sizes:
            step_id = ('TEST_SIZE', emb_size, hid_size)
            if step_id in completed:
                print(f"⏩ Skipping {step_id}, already completed.")
                continue

            config = best_config.copy()
            config['emb_size'] = emb_size
            config['hid_size'] = hid_size
            print(f"\n🚀 [TEST_SIZE] Running: emb={emb_size}, hid={hid_size} - {i+1}/{total_combinations}")
            final_ppl = run_experiment(
                config=config,
                weight_tying=config['weight_tying'],
                variational_dropout=config['variational_dropout'],
                nt_avsgd=config['nt_avsgd'],
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                test_dataset=test_dataset,
                lang=lang
            )

            if final_ppl < best_ppl:
                best_ppl = final_ppl
                best_config['emb_size'] = emb_size
                best_config['hid_size'] = hid_size

            state['completed'].append(step_id)
            state['best_config'] = best_config
            state['best_ppl'] = best_ppl
            save_state(state)
            i += 1

    if TEST_BATCH:
        batch_sizes = [32, 128]
        total_combinations = len(batch_sizes)
        i = 0

        for batch_size in batch_sizes:
            step_id = ('TEST_BATCH', batch_size)
            if step_id in completed:
                print(f"⏩ Skipping {step_id}, already completed.")
                continue

            config = best_config.copy()
            config['train_batch'] = batch_size
            print(f"\n🚀 [TEST_BATCH] Running: batch_size={batch_size} - {i+1}/{total_combinations}")
            run_experiment(
                config=config,
                weight_tying=config['weight_tying'],
                variational_dropout=config['variational_dropout'],
                nt_avsgd=config['nt_avsgd'],
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                test_dataset=test_dataset,
                lang=lang
            )

            state['completed'].append(step_id)
            save_state(state)
            i += 1

    print("\n🏁 Esperimenti completati.")
    print(f"🔧 Miglior configurazione finale:\n{best_config}")
    print(f"📉 Perplexity: {best_ppl}")
