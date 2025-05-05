from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from bert_model import BERTIntentSlot
from utils import BertAtisDataset, bert_collate_fn
from utils import load_ATIS, create_dev_set, Lang, device
import torch
import torch.nn as nn

# Load data and tokenizer
tmp_train_raw, test_raw = load_ATIS()
train_raw, dev_raw, test_raw = create_dev_set(tmp_train_raw, test_raw)

words = sum([x['utterance'].split() for x in train_raw], [])
corpus = train_raw + dev_raw + test_raw
slots = set(sum([line['slots'].split() for line in corpus], []))
intents = set([line['intent'] for line in corpus])
lang = Lang(words, intents, slots, cutoff=0)

# Tokenizer and Dataset
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
test_dataset = BertAtisDataset(test_raw[:4], tokenizer, lang, max_len=64)
test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=bert_collate_fn)

# Model
model = BERTIntentSlot("bert-base-uncased", len(lang.intent2id), len(lang.slot2id)).to(device)
model.eval()

# Single batch test
with torch.no_grad():
    for batch in test_loader:
        slot_logits, intent_logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids']
        )

        print("Intent logits:", intent_logits.shape)  # (B, C_intent)
        print("Slot logits:", slot_logits.shape)      # (B, T, C_slot)
        break
