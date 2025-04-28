import torch
import torch.nn as nn
from transformers import BertModel


# * implementation inspried by: https://github.com/monologg/JointBERT/blob/master/model/modeling_jointbert.py

class BertForJointIntentAndSlot(nn.Module):
    def __init__(self, model_name, num_intent_labels, num_slot_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.intent_classifier = nn.Linear(hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(hidden_size, num_slot_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        seq_out = self.dropout(outputs.last_hidden_state)
        pool_out = self.dropout(outputs.pooler_output)
        intent_logits = self.intent_classifier(pool_out)
        slot_logits = self.slot_classifier(seq_out)
        return intent_logits, slot_logits