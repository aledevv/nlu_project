import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

#* Variational Dropout (took inspiration from: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html#Variational_Dropout)
class VariationalDropout(nn.Module):
    def __init__(self, dropout):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout).expand_as(x)
        return x * mask

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, use_weight_tying=False, use_variational_dropout=False):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        
        self.use_weight_tying = use_weight_tying
        self.use_variational_dropout = use_variational_dropout   
        
        #* Weight tying: https://github.com/pytorch/examples/blob/main/word_language_model/model.py#L31
        if self.use_weight_tying and hidden_size != emb_size:   #? If hidden size doesn't match embedding size we need to perform a (linear) mapping
            self.hid2emb = nn.Linear(hidden_size, emb_size)
            output_input_dim = emb_size
        else:                                                   #? if hidden_seze == emb_size: no transformation
            self.hid2emb = None
            output_input_dim = hidden_size
        
        #? weight tying is applied to last layer (because softmax is implicitly applied making CrossEntropyLoss, and so it is not explicitated here)
        
        self.output = nn.Linear(output_input_dim, output_size)
        
        if self.use_weight_tying:
            print("Applying weight tying")
            self.output.weight = self.embedding.weight
            
        if self.use_variational_dropout:
            print("Applying variational dropout")
            self.dropout_emb = VariationalDropout(emb_dropout)
            self.dropout_out = VariationalDropout(out_dropout)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        
        if self.use_variational_dropout:            #? Apply dropouts in the same points as in Part A (one after the embedding and another one before the output)
            emb = emb = self.dropout_emb(emb)
        
        lstm_out, _  = self.lstm(emb)
        
        if self.use_variational_dropout:
            lstm_out = self.dropout_out(lstm_out)
        
        #? applying weight tying if active
        if self.use_weight_tying and self.hid2emb is not None:  #? apply the mapping due to hid_size != emb_size
            logits = self.output(self.hid2emb(lstm_out))
        else:
            logits = self.output(lstm_out)
            
        return logits.permute(0, 2, 1)