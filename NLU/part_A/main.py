# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file    
from functions import *


# 📦 Experiments configuration
base = {
    'batch_size_train': 128,
    'batch_size_eval': 64,
    'n_epochs': 200,
    'clip': 5,
    'runs': 3,
    'patience': 3,
    'cutoff': 0,
}

# Specific configs for the experiments
experiments = [
    {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 1e-4, **flags}
    for flags in [
        # {'bidirectional': False, 'dropout': False, 'dropout_rate': 0.0, 'n_layers': 1}, # Vanilla
        {'bidirectional': True, 'dropout': False, 'dropout_rate': 0.0, 'n_layers': 1},  # Bidirectional
        {'bidirectional': False, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1},  # Just dropout
        {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1},   # bidirecitional and dropout
        {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.3, 'n_layers': 1},   # same but higher dropout probability
        {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 2},   # trying 2 layers with both modifications
        {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1, 'hid_size': 300}, # both but with higher num of layers
        {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1, 'emb_size': 400}, # both but higher embedding size
    ]
]


if __name__ == "__main__":
    
    # Initialize the list of results
    all_results = []
    experiment_idx=0
    
    for cfg in experiments:
        
        print(f"=== 🏁 Started experiment {experiment_idx+1} of {len(experiments)} ===")
        
        lang, train_dataset, dev_dataset, test_dataset = prepare_data(cfg)
        train_loader, dev_loader, test_loader = get_dataloaders(train_dataset, dev_dataset, test_dataset, cfg)

        slot_f1s, intent_accs, all_tr, all_dev, all_ep = run_experiments(
            config=cfg,
            model_class=ModelIAS,
            data_loaders=(train_loader, dev_loader, test_loader),
            lang=lang,
        )
        
        experiment_idx+=1


