# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file    
from functions import *


# 📦 Experiments configuration
base = {
    "bert_model": "bert-base-uncased",
    "max_len": 64,
    'batch_size_train': 128,
    'batch_size_eval': 64,
    'n_epochs': 200,
    'clip': 5,
    'runs': 1,
    'patience': 3,
    'cutoff': 0,
}

# Specific configs for the experiments
experiments = [
    {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 1e-4,}
    for flags in [
        {}, # Vanilla
        # {'bidirectional': True, 'dropout': False, 'dropout_rate': 0.0, 'n_layers': 1},  # Bidirectional
        # {'bidirectional': False, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1},  # Just dropout
        # {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1},   # bidirecitional and dropout
        # {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.3, 'n_layers': 1},   # same but higher dropout probability
        # {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 2},   # trying 2 layers with both modifications
        # {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1, 'hid_size': 300}, # both but with higher num of layers
        # # {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1, 'emb_size': 400}, # both but higher embedding size
        # {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.8, 'n_layers': 1},   # test with critically higher dropout
        # {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 5},   # trying 5 layers with both modifications
        # {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 10},   # trying 10 layers with both modifications
        # {'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1, 'hid_size': 300, 'emb_size': 400}, # both enhancement in hid and emb size
        


        # # ? Extra experiments
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 1e-3, 'batch_size_train': 64, 'batch_size_eval': 64, 'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1},
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 5e-4, 'batch_size_train': 64, 'batch_size_eval': 64, 'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1},
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 3e-4, 'batch_size_train': 256, 'batch_size_eval': 128, 'bidirectional': False, 'dropout': False, 'dropout_rate': 0.0, 'n_layers': 1},
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 1e-4, 'batch_size_train': 64, 'batch_size_eval': 64, 'bidirectional': True, 'dropout': False, 'dropout_rate': 0.0, 'n_layers': 1},
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 5e-5, 'batch_size_train': 128, 'batch_size_eval': 64, 'bidirectional': True, 'dropout': True, 'dropout_rate': 0.3, 'n_layers': 1},
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 5e-4, 'batch_size_train': 256, 'batch_size_eval': 128, 'bidirectional': False, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1},
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 3e-4, 'batch_size_train': 128, 'batch_size_eval': 64, 'bidirectional': True, 'dropout': False, 'dropout_rate': 0.0, 'n_layers': 1},
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 5e-5, 'batch_size_train': 64, 'batch_size_eval': 64, 'bidirectional': False, 'dropout': True, 'dropout_rate': 0.3, 'n_layers': 1},
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 1e-4, 'batch_size_train': 256, 'batch_size_eval': 128, 'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1},

        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 0.01, 'batch_size_train': 128, 'batch_size_eval': 64, 'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1}, # high lr
        # {**base, 'hid_size': 200, 'emb_size': 300, 'lr': 1e-6, 'batch_size_train': 128, 'batch_size_eval': 64, 'bidirectional': True, 'dropout': True, 'dropout_rate': 0.1, 'n_layers': 1}, # very small lr

    ]
]


if __name__ == "__main__":
    
    if len(experiments) == 0: #! SBLOCCA
        print("NO experiments set")
        quit()
    
    # Initialize the list of results
    all_results = []
    experiment_idx=0
    
    
    
    for cfg in experiments:
        
        print(f"=== 🏁 Started experiment {experiment_idx+1} of {len(experiments)} ===")
        
        # * DATA SETUP
        train_raw, dev_raw, test_raw = prepare_data(cfg)
        
        # Tokenizer BERT
        tokenizer = load_tokenizer(base['bert_model'])

        # Costruisci mapping intent e slot
        intent2id, slot2id = get_label_maps(train_raw + dev_raw + test_raw)

        # Crea dataset BERT
        train_dataset = BERTJointDataset(train_raw, tokenizer, intent2id, slot2id, max_len=cfg['max_len'])
        dev_dataset   = BERTJointDataset(dev_raw,   tokenizer, intent2id, slot2id, max_len=cfg['max_len'])
        test_dataset  = BERTJointDataset(test_raw,  tokenizer, intent2id, slot2id, max_len=cfg['max_len'])
        
        train_loader, dev_loader, test_loader = create_data_loaders(cfg, train_dataset, dev_dataset, test_dataset)

        slot_f1s, intent_accs, all_tr, all_dev, all_ep = run_experiments(
            config=cfg,
            model_class=BertForJointIntentAndSlot,
            data_loaders=(train_loader, dev_loader, test_loader),
            lang=lang,
        )
        
        experiment_idx+=1



