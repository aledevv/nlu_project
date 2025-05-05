from functions import *
from utils import bert_collate_fn
from bert_model import BERTIntentSlot

# 📦 Experiments configuration
base = {
    'batch_size_train': 16,
    'batch_size_eval': 32,
    'n_epochs': 10,
    'clip': 1,
    'runs': 1,
    'patience': 2,
    'cutoff': 0,
}

experiments = [
    {**base, 'lr': 5e-5}
]

if __name__ == "__main__":
    if len(experiments) == 0:
        print("NO experiments set")
        quit()

    all_results = []
    experiment_idx = 0

    for cfg in experiments:
        print(f"=== 🏁 Started experiment {experiment_idx+1} of {len(experiments)} ===")

        lang, train_dataset, dev_dataset, test_dataset = prepare_data(cfg)

        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size_train'], shuffle=True, collate_fn=bert_collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=cfg['batch_size_eval'], collate_fn=bert_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size_eval'], collate_fn=bert_collate_fn)

        def model_factory():
            return BERTIntentSlot(
                model_name="bert-base-uncased",
                num_intents=len(lang.intent2id),
                num_slots=len(lang.slot2id)
            ).to(device)

        model = model_factory()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
        criterion_slots = nn.CrossEntropyLoss(ignore_index=lang.slot2id['pad'])
        criterion_intents = nn.CrossEntropyLoss()

        slot_f1s, intent_accs, all_tr, all_dev, all_ep = run_experiments(
            config=cfg,
            model_class=model_factory,
            data_loaders=(train_loader, dev_loader, test_loader),
            lang=lang,
        )

        experiment_idx += 1