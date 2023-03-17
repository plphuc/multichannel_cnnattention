from CNNModel import CNNModel
from SelfAttentionModel import AttentionModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MultiChannel_CNNAttentionModel(nn.Module):
    def __init__(self, bert, output_dim, dropout, n_filters, filter_sizes, batch_size, hidden_dim, vocab_size, embedding_length):
        super().__init__()
        self.bert = bert
        self.output_dim = output_dim
        self.dropout = dropout
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.batch_size = batch_size
        self.hidden_size = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.cnn = CNNModel(bert, output_dim, dropout, n_filters, filter_sizes)
        self.attention = AttentionModel(bert, batch_size, output_dim, hidden_dim, vocab_size, embedding_length)
        
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(400, output_dim)
        self.do = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, text, batch_size):
        with torch.no_grad():
            input = self.bert(text)[0]
        cnn_output = self.cnn(input)
        attention_output = self.attention(input, batch_size)
        cnn_attention_cat = torch.cat((cnn_output, attention_output), 1)
        output_ln1 = self.fc2(cnn_attention_cat)
        output_do = self.do(output_ln1)
        # output_ln2 = self.fc2(output_do)

        return self.softmax(output_do)