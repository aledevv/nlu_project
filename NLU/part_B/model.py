import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class BertForIntentAndSlot(BertPreTrainedModel):
    def __init__(self, config, num_intent_labels, num_slot_labels):
        super().__init__(config)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels)
        self.num_intent_labels = num_intent_labels  # Store the number of intent labels
        self.num_slot_labels = num_slot_labels    # Store the number of slot labels
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, intent_labels=None, slot_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # Intent classification
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        # Slot filling
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        loss = None
        if intent_labels is not None and slot_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_intent = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))
            loss_slot = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1))
            loss = loss_intent + loss_slot

        output = (intent_logits, slot_logits) + outputs[2:]
        return ((loss,) + output) if loss is not None else output