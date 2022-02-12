import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import copy

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved. 
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.append(root_dir)
from xai_transformer import make_p_layer, BertPooler, BertAttention


from imdb import load_imdb, MovieReviewDataset, create_data_loader

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

MAX_LEN = 512
BATCH_SIZE = 32
RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

df = load_imdb()
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


torch.cuda.empty_cache()
class_names = ['negative', 'positive']

#model = SentimentClassifier()


class Config(object):
    
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.layer_norm_eps = 1e-12
        self.n_classes = 2
        self.n_blocks = 3
                    
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.detach_layernorm = False # Detaches the attention-block-output LayerNorm
        self.detach_kq = False # Detaches the kq-softmax branch
        self.device = device
        self.train_mode = True
        
        
from transformers import BertForSequenceClassification

bert_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
bert_model.bert.embeddings.requires_grad = False
for name, param in bert_model.named_parameters():                
    if name.startswith('embeddings'):
        param.requires_grad = False
        
pretrained_embeds = bert_model.bert.embeddings


config = Config()
model = BertAttention(config, pretrained_embeds)
model.to(device)

print('\n\n\n\n')

print(model)
EPOCHS = 20

# optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

params_to_train = {k:v for k,v in model.named_parameters() if 'embed' not in k}
ignore_params = [k for k,v in model.named_parameters() if 'embed' in k]

optimizer = AdamW(params_to_train.values(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples,
):
  model = model.train()

  losses = []
  correct_predictions = 0
  import torch.nn.functional as F

  current_loss = 0.0
  for i , d in enumerate(tqdm(data_loader)):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    token_type_ids = d["token_type_ids"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      token_type_ids=token_type_ids,
      attention_mask=attention_mask,
      labels = targets
    )['logits']
  

    preds = outputs.argmax(dim=1) #torch.max(outputs, dim=1)
    loss = F.cross_entropy(outputs , targets) # , label_smoothing=0.4)#loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    current_loss += loss.item()

    if np.isnan(current_loss):
        nan_ids =  np.argwhere(np.isnan(outputs.sum(1).detach().cpu().numpy())).squeeze().tolist()
        if  isinstance(nan_ids, int):
            nan_ids = [nan_ids]
        
        for id_ in nan_ids:
            print(tokenizer.convert_ids_to_tokens(input_ids[id_]), targets[id_])
        import pdb;pdb.set_trace()
    
    if i % 100 == 0 and i != 0:
        
        
        print('Training Loss is {}'.format(str(current_loss / i)))
    nn.utils.clip_grad_norm_(params_to_train.values(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )['logits']

      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

history = defaultdict(list)
best_accuracy = 0
from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()

for epoch in tqdm(range(EPOCHS)):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    test_data_loader,
    loss_fn, 
    device, 
    len(df_test)
  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state_dropout_{:0.3f}.bin'.format(val_acc))
    best_accuracy = val_acc

