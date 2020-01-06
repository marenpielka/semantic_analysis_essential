from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


MAX_LENGTH = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, pretrained=None, bidirectional=False, freeze_wv=True):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.dim = 2 if bidirectional else 1

        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained).float(), freeze=freeze_wv)
            embedding_size = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=bidirectional)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.dim, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size=100, pretrained=None, bidirectional = False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dim = 2 if bidirectional else 1

        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained).float(), freeze=False)
            embedding_size = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=bidirectional)
        #self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional)

        self.out = nn.Linear(self.dim*hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = F.relu(embedded)
        output, hidden = self.gru(embedded, hidden)
        #output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (torch.zeros(self.dim, 1, self.hidden_size, device=device))#,
                #torch.zeros(self.dim, 1, self.hidden_size, device=device))


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, pretrained=None):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained).float(), freeze=True)
            embedding_size = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(output_size, embedding_size)
        self.attn = nn.Linear(embedding_size + hidden_size, self.max_length)
        self.attn_combine = nn.Linear(embedding_size + hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, apply_attn=True):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        if apply_attn:
            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                     encoder_outputs.unsqueeze(0))

            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)

        else:
            output = embedded

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

