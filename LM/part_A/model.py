import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
        
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.dopout1 = nn.Dropout(emb_dropout)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.dopout2 = nn.Dropout(out_dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop_out = self.dopout1(emb)
        rnn_out, _  = self.lstm(drop_out)
        drop_out2 = self.dopout2(rnn_out)
        output = self.output(drop_out2).permute(0,2,1)
        return output