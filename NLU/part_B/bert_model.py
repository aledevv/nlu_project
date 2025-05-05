import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast

class BERTIntentSlot(nn.Module):
    def __init__(self, model_name: str, num_intents: int, num_slots: int):
        super(BERTIntentSlot, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask, token_type_ids, slot_label_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state       # (B, T, H)
        pooled_output = outputs.pooler_output             # (B, H)

        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)

        intent_logits = self.intent_classifier(pooled_output)       # (B, C_intent)
        slot_logits = self.slot_classifier(sequence_output)         # (B, T, C_slot)

        return slot_logits, intent_logits
