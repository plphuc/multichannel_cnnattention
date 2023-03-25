# $ git clone https://github.com/plphuc/multichannel_cnnattention.git

# /content/drive/MyDrive/Yelp_dataset/Processed_Yelp.csv

# !nvidia-smi

# !pip3 install virtualenv
# !virtualenv theanoEnv
# Create and start a virtual environment
# !virtualenv -p python3 env 
# !source env/bin/activate

# !pip3 install -r /content/multichannel_cnnattention/requirements.txt
# !pip install transformers

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/multichannel_cnnattention
import random as rd
import numpy as np
import time

import torch
from torchtext import datasets, data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from transformers import BertTokenizer, AutoTokenizer, XLMRobertaTokenizer, BertModel
import sys

data_path = "./data/"
train_name = "train.csv"
test_name = "test.csv"
model_save_path = sys.argv[1]



SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print('Bert Tokenizer Loaded...')

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

BATCH_SIZE = 64
max_input_length = 400

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

TEXT = data.Field(batch_first=True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField()
fields = [('text',TEXT),('sentiment',LABEL)]

train_data, test_data = data.TabularDataset.splits(
                            path = data_path,
                            train = train_name,
                            test = test_name,
                            format = 'csv',
                            fields = fields,
                            skip_header = True)

train_data, valid_data = train_data.split(random_state = rd.seed(SEED))
print('Data loading complete')
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of test examples: {len(test_data)}")

LABEL.build_vocab(train_data, valid_data)
print(LABEL.vocab.stoi)
TEXT.build_vocab(train_data, valid_data)
VOCAB_SIZE = len(TEXT.vocab.stoi)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device in use: ", device)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                  (train_data, valid_data, test_data),
                                                  sort_key = lambda x: len(x.text),
                                                  batch_size = BATCH_SIZE,
                                                  device = device)

from transformers import BertModel
bert = BertModel.from_pretrained('bert-base-uncased')

def categorical_accuracy(preds, y):
  count0, count1 = torch.zeros(1).to(device), torch.zeros(1).to(device)
  max_preds = preds.argmax(dim = 1, keepdim = True)
  correct = max_preds.squeeze(1).eq(y) # True_False matrix
  predictions = max_preds.squeeze(1) # Predicted class
  true_correct = [0,0]
  y_np = y.detach().cpu().numpy()
  for i, j in enumerate(y_np):
    true_correct[j] += 1 #count label of each class
    if j == 0:
      count0 += correct[j] #count True
    elif j == 1:
      count1 += correct[j]
  metric = torch.FloatTensor([count0/true_correct[0], count1/true_correct[1], f1_score(y_np, predictions.detach().cpu().numpy(), average = 'macro')])
  return correct.sum()/torch.FloatTensor([y.shape[0]]).to(device), metric, confusion_matrix(y.detach().cpu().numpy(),max_preds.detach().cpu().numpy())
  #acc, all_acc, confusion_matrix

def clip_gradient(model, clip_value):
  params = list(filter(lambda p: p.grad is not None, model.parameters()))
  for p in params:
      p.grad.data.clamp_(-clip_value, clip_value)

def train(model, iterator, optimizer, criterion):
  epoch_loss = 0
  epoch_acc = 0

  model.train()
  for batch in iterator:

    # Resets the gradients after every batch
    optimizer.zero_grad() 

    predictions = model(batch.text, batch_size = len(batch.text))
    # Compute the loss
    loss = criterion(predictions, batch.sentiment)
    acc,_,_ = categorical_accuracy(predictions,batch.sentiment)

    # Backpropage the loss and compute the gradients
    loss.backward()
    clip_gradient(model, 1e-1)

    # Update the weight
    optimizer.step()

    epoch_loss += loss.item()
    epoch_acc += acc.item()
  return epoch_loss/len(iterator), epoch_acc/len(iterator)

def evaluate(model, iterator, criterion):
  epoch_loss = 0
  epoch_acc = 0
  epoch_all_acc = torch.FloatTensor([0,0,0]) #metric = [acc0, acc1, f1_score]
  confusion_mat = torch.zeros((2,2))
  confusion_mat_temp = torch.zeros((2,2))
  model.eval()

  with torch.no_grad():
    for batch in iterator:
      predictions = model(batch.text, batch_size = len(batch.text))

      loss = criterion(predictions, batch.sentiment)
      acc,all_acc,confusion_mat_temp = categorical_accuracy(predictions, batch.sentiment)
      
      epoch_loss += loss.item()
      epoch_acc += acc.item()
      epoch_all_acc += all_acc
      confusion_mat+=confusion_mat_temp
    return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_all_acc/len(iterator),confusion_mat
    # valid_loss, valid_acc, tot, conf

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parmeters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""# Check params"""

OUTPUT_DIM = 2
DROPOUT = 0.5
N_FILTERS = 100
FILTER_SIZES = [2,3,4]
HIDDEN_DIM = 100

from MultiChannel_CNNAttentionModel import MultiChannel_CNNAttentionModel

# model = torch.load(model_save_name)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

# from MultiChannel_CNNAttentionModel import MultiChannel_CNNAttentionModel
# model_name = "cnn_attention_model2"
model = MultiChannel_CNNAttentionModel(bert, OUTPUT_DIM, DROPOUT, N_FILTERS, FILTER_SIZES, BATCH_SIZE, HIDDEN_DIM, VOCAB_SIZE, 768)

model = model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss().to(device)
nll_loss = nn.NLLLoss().to(device)
log_softmax = nn.LogSoftmax().to(device)

"""# Execute model"""

N_EPOCHS = 40
best_f1 = -1

# Set model in training phase
for epoch in range(N_EPOCHS):
  start_time = time.time()
  train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
  valid_loss, valid_acc, tot, conf = evaluate(model, valid_iterator, criterion)
  f1 = tot[2]
  end_time = time.time()

  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  
  if f1 > best_f1:
    best_f1 = f1
    torch.save(model, model_save_path)

  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
  print(tot)
  print(conf)

def evaluate_save_to_file (model, test_iterator):
  label_dict = {'0': "negative", '1': "positive"}
  file = open("/content/answer.txt", "w")
  file.write('uid,sentiment\n')
  count = 0
  for batch in test_iterator:
    predictions = model(batch.text, batch_size=len(batch)).squeeze(1)
    # predictions = model(batch.text).squeeze(1)
    max_preds = predictions.argmax(dim=1, keepdim=True).detach().cpu().numpy()
    for i, row in enumerate(batch.sentiment.cpu().numpy()):
      count += 1
      label_number = max_preds[i][0]
      label_number_str = list(LABEL.vocab.stoi.keys())[list(LABEL.vocab.stoi.values()).index(label_number)]
      predicted_label_name = label_dict[label_number_str]
      if count != len(test_data):
        file.write('%s,%s\n'%(i,predicted_label_name))
      else:
        file.write('%s,%s'%(i,predicted_label_name))
  file.close()

evaluate_save_to_file(model, test_iterator)

valid_loss, valid_acc, tot, conf = evaluate(model, test_iterator, criterion)
print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
print(tot)