from functions import *
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import BertTokenizerFast # Import BertTokenizer

# 📦 Experiments configuration
base = {
    'batch_size_train': 64,  # Reduced batch size for BERT
    'batch_size_eval': 128,
    'n_epochs': 50,         # Adjust as needed
    'clip': 1,              # Gradient clipping
    'runs': 1,              # Number of training runs
    'patience': 3,          # Patience for early stopping
    'cutoff': 0,            # Cutoff for rare words (if used)
    'lr': 1e-4,             # Learning rate (crucial for BERT)
    'model_name': 'bert-base-uncased',  # Specify the BERT model
    # --- BERT-specific parameters (you can add more if needed) ---
    'bert_hidden_size': 768,  # Hidden size of BERT-base (adjust for -large)
    'bert_dropout_prob': 0.1,  # Dropout probability in BERT
    'bert_max_len': 50       # Max sequence length for BERT
}

# Specific configs for the experiments
experiments = [
    {**base, 'lr': 2e-5, 'model_name': 'bert-base-uncased'} # Experiment with learning rate and model size
    # Add more experiments as needed (e.g., different BERT models, learning rates)
]


if __name__ == "__main__":

    if len(experiments) == 0:
        print("NO experiments set")
        quit()

    # Initialize the list of results
    all_results = []
    experiment_idx = 0

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Or bert-large-uncased
    
    lang, train_dataset, dev_dataset, test_dataset = prepare_data(cfg, tokenizer)  # Get tokenizer

    for cfg in experiments:

        print(f"=== 🏁 Started experiment {experiment_idx + 1} of {len(experiments)} ===")

        train_loader, dev_loader, test_loader = get_dataloaders(train_dataset, dev_dataset, test_dataset, cfg)

        slot_f1s, intent_accs, all_tr, all_dev, all_ep = run_experiments(
            config=cfg,
            model_class=BertForIntentAndSlot,  # Use the BERT model class
            data_loaders=(train_loader, dev_loader, test_loader),
            lang=lang,
            tokenizer=tokenizer # Pass the tokenizer
        )

        experiment_idx += 1