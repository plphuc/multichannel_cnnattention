import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1234)

class CNNModel(nn.Module):
    def __init__(self, bert, output_dim, dropout, n_filters, filter_sizes):
        super().__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.conv_0 = nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(1, n_filters, (filter_sizes[2], embedding_dim))
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = input.unsqueeze(1)

        #embedded = [batch size, 1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
        return cat
