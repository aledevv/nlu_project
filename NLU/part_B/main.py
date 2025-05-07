from functions import *
from utils import bert_collate_fn
from model import BERTIntentSlot
from transformers.configuration_utils import PretrainedConfig
import itertools
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 💼 Base config
base = {
    'n_epochs': 10,
    'clip': 1,
    'runs': 3,
    'patience': 3,
    'cutoff': 0,
}

# 🔍 Grid of values
param_grid = {
    'lr': [1e-5, 3e-5, 5e-5, 1e-4],
    'dropout': [0.1, 0.2, 0.3, 0.4],
}

batch_configs = [
    {'batch_size_train': 64, 'batch_size_eval': 128},
    {'batch_size_train': 32, 'batch_size_eval': 64},
    {'batch_size_train': 16, 'batch_size_eval': 32},
]

param_names = list(param_grid.keys())
param_combinations = list(itertools.product(*param_grid.values()))

if __name__ == "__main__":
    for batch_cfg in batch_configs:
        print(f"\n=== 🚀 Running grid for train={batch_cfg['batch_size_train']} eval={batch_cfg['batch_size_eval']} ===")

        experiments = [
            {**base, **batch_cfg, **dict(zip(param_names, values))}
            for values in param_combinations
        ]

        all_results = []
        experiment_idx = 0

        for cfg in experiments:
            print(f"=== 🏁 Experiment {experiment_idx+1}/{len(experiments)} ===")
            print("Config:", cfg)

            lang, train_dataset, dev_dataset, test_dataset = prepare_data(cfg)

            train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size_train'], shuffle=True, collate_fn=bert_collate_fn)
            dev_loader = DataLoader(dev_dataset, batch_size=cfg['batch_size_eval'], collate_fn=bert_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size_eval'], collate_fn=bert_collate_fn)

            def model_factory():
                return BERTIntentSlot(
                    model_name="bert-base-uncased",
                    num_intents=len(lang.intent2id),
                    num_slots=len(lang.slot2id),
                    dropout_prob=cfg['dropout'],
                ).to(device)

            model = model_factory()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
            criterion_slots = nn.CrossEntropyLoss(ignore_index=lang.slot2id['pad'])
            criterion_intents = nn.CrossEntropyLoss()

            slot_f1s, intent_accs, all_tr, all_dev, all_ep = run_experiments(
                config=cfg,
                model_class=model_factory,
                data_loaders=(train_loader, dev_loader, test_loader),
                lang=lang,
            )

            score = slot_f1s.mean()+intent_accs.mean()

            all_results.append({
                'lr': cfg['lr'],
                'dropout': cfg['dropout'],
                'score': round(score, 4),
            })
            experiment_idx += 1

        # Save heatmap
        df = pd.DataFrame(all_results)
        heatmap_data = df.pivot(index="dropout", columns="lr", values="score")

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Slot F1 + Intent Acc'})
        plt.title(f"Slot F1 - Grid Search (Train={batch_cfg['batch_size_train']} Eval={batch_cfg['batch_size_eval']})")
        plt.xlabel("Learning Rate")
        plt.ylabel("Dropout")
        plt.tight_layout()

        os.makedirs("experiments", exist_ok=True)
        img_path = f"experiments/grid_f1_train{batch_cfg['batch_size_train']}_eval{batch_cfg['batch_size_eval']}.png"
        plt.savefig(img_path)
        print(f"📈 Saved heatmap: {img_path}")
