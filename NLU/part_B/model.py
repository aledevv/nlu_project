import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.configuration_utils import PretrainedConfig

class BertForIntentAndSlot(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, num_intent_labels, num_slot_labels):
        super(BertForIntentAndSlot, self).__init__(config)
        self.bert = BertModel(config)
        
        # # Freeze BERT parameters if needed to make it a feature extractor (faster)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        self.num_intent_labels = num_intent_labels  # Store the number of intent labels
        self.num_slot_labels = num_slot_labels    # Store the number of slot labels
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # Intent classification
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        # Slot filling
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits