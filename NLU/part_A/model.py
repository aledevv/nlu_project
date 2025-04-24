import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1,
                 bidirectional=False, dropout=False, dropout_rate=0.1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)
        
        encoder_dim = hid_size * (2 if bidirectional else 1) # num of layers has to double in case of bidir   
        self.slot_out = nn.Linear(encoder_dim, out_slot)
        self.intent_out = nn.Linear(encoder_dim, out_int)
   
        # if flag off -> Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout else nn.Identity()
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        utt_emb = self.dropout(utt_emb) #! DROPOUT LAYER
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        # h_n: (num_layers * num_directions, batch, hid_size)
        if self.bidirectional:
            # concat ultima layer forward e backward
            h_forward = last_hidden[-2, :, :]   # ultimo layer, forward
            h_backward = last_hidden[-1, :, :]  # ultimo layer, backward
            last_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            last_hidden = last_hidden[-1, :, :]
        
        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent