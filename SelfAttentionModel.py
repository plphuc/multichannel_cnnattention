import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class AttentionModel(torch.nn.Module):
	def __init__(self, bert, batch_size, output_dim, hidden_dim, vocab_size, embedding_length):
		super(AttentionModel, self).__init__()
		
		self.batch_size = batch_size
		self.output_size = output_dim
		self.hidden_size = hidden_dim
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.bert = bert
		# self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
		self.lstm = nn.LSTM(embedding_length, hidden_dim)
		self.fc = nn.Linear(hidden_dim, output_dim)
		
	def attention_net(self, lstm_output, final_state):
		
		hidden = final_state.squeeze(0)
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
		return new_hidden_state
	
	def forward(self, input, batch_size):
		input = input.permute(1, 0, 2)
		h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			
		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 
		output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
		
		attn_output = self.attention_net(output, final_hidden_state)
		# logits = self.fc(attn_output)
		
		return attn_output