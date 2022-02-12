from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import copy
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainerDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, tokenizer):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer

        # Tokenize the input
        self.tokenized_inputs = tokenizer(inputs, padding=False)   

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        features = {'input_ids':   self.tokenized_inputs['input_ids'][idx],
                   'token_type_ids':self.tokenized_inputs['token_type_ids'][idx],
                   'attention_mask':self.tokenized_inputs['attention_mask'][idx],
                   'label':self.targets[idx]
                   }
        return features
    
    
def get_sst_dataset(datasets, tokenizer):


    # Load datasets
    train_dataset = TrainerDataset(datasets["train"]["sentence"],
                                   datasets["train"]["label"], tokenizer)

    eval_dataset = TrainerDataset(datasets["validation"]["sentence"],
                                  datasets["validation"]["label"], tokenizer)


    return train_dataset, eval_dataset
