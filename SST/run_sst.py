import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from xai_transformer import make_p_layer, BertPooler, BertAttention
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sst import get_sst_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric
from attribution import softmax, compute_joint_attention, get_flow_relevance_for_all_layers , _compute_rollout_attention
from utils import flip, set_up_dir
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import matplotlib.pyplot as plt
import pickle

RANDOM_SEED = 42


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

# Load data
datasets  = load_dataset("glue", 'sst2')


label_to_id = {v: i for i, v in enumerate([0,1])}

_, test_data_loader = get_sst_dataset(datasets, tokenizer)


# Init Model
class Config(object):
    
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.layer_norm_eps = 1e-12
        self.n_classes = 2
        self.n_blocks = 3
                    
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.detach_layernorm = True # Detaches the attention-block-output LayerNorm
        self.detach_kq = True # Detaches the kq-softmax branch
        self.device = device
        self.train_mode = False
        self.detach_mean = True #

        

bert_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
bert_model.bert.embeddings.requires_grad = False
for name, param in bert_model.named_parameters():                
    if name.startswith('embeddings'):
        param.requires_grad = False
        
pretrained_embeds = bert_model.bert.embeddings

params = torch.load('SST/sst2-3layer-model.pt', map_location=torch.device(device))

def rename_params(key):
    for k_ in ['key','query', 'value']:
        key=key.replace(k_, 'p'+k_)
    return key


# Transformer Model 
config = Config()
config.detach_layernorm = False # Detaches the attention-block-output LayerNorm
config.detach_kq = False
model = BertAttention(config, pretrained_embeds)
model.load_state_dict(params)
model.to(device)
models = {'none': model}

# Transformer Model 
print('Detach SM')
config = Config()
config.detach_layernorm = False # Detaches the attention-block-output LayerNorm
config.detach_kq = True
model = BertAttention(config, pretrained_embeds)
model.load_state_dict(params)
model.to(device)
models['detach_KQ'] = model


print('Detach SM +  LN')
# Transformer Model 
config = Config()
model = BertAttention(config, pretrained_embeds)
model.load_state_dict(params)
model.to(device)
models['detach_KQ_LNorm'] = model

print('Detach SM +  LN without mean')
# Transformer Model 
config = Config()
config.detach_layernorm = True # Detaches the attention-block-output LayerNorm
config.detach_mean = False # Detaches the attention-block-output LayerNorm
config.detach_kq = True
model = BertAttention(config, pretrained_embeds)
model.load_state_dict(params)
model.to(device)
models['detach_KQ_LNorm_Norm'] = model





# Run Flipping
UNK_token = tokenizer.unk_token_id

fracs = np.linspace(0.,1.,11)

print(len(test_data_loader))
# Uncomment cases if they are not computed already in all_flips
gammas = [0.00,0.00, 0.00]
for flip_case in [ 'generate' , 'pruning']:
    print(flip_case)
    all_flips = {}
    
    res_conservation = {}
    for case, random_order in [('random', True), 
                               ('attn_last', False),  
                               ('rollout_2', False),
                               ('GAE' , False),
                               ('gi', False),
                               ('lrp_detach_KQ', False),
                               ('lrp_detach_KQ_LNorm_Norm', False),
                              ]:
        print(case)
        layer_idxs = model.attention_probs.keys()
        M,E, EVOLUTION = [],[], []
        C = []
        j=0
        if case in ['gi', 'lrp', 'GAE']:
            model = models['none']
        elif case in ['gi_detach_KQ', 'lrp_detach_KQ']:
            model = models['detach_KQ']
        elif case in ['detach_KQ_LNorm_Norm', 'lrp_detach_KQ_LNorm_Norm']   : 
            model = models['detach_KQ_LNorm_Norm']
        else:
            model = models['detach_KQ_LNorm']
        for x in test_data_loader:
            input_ids = torch.tensor(np.float32(x['input_ids']) , requires_grad=True).unsqueeze(0).long().to(device)
            attention_mask = torch.tensor(np.float32(x['attention_mask']), requires_grad=True).unsqueeze(0).long().to(device)
            token_type_ids = torch.tensor(np.float32(x['token_type_ids'])).unsqueeze(0).long().to(device)
            words = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
            y_true = torch.tensor(x['label']).to(device)
            labels_in = torch.tensor([int(y_true)]*len(input_ids)).long().to(device)
            if case == 'GAE':
                outs = model.forward_and_explain(input_ids=input_ids, cl=y_true,
                 labels = labels_in , method=case)
            else:
                outs = model(input_ids=input_ids,
                             labels = labels_in)
            loss = outs['loss'].detach().cpu().numpy()
            y_pred = np.argmax(outs['logits'].squeeze().detach().cpu().numpy())
            if case =='random':
                attribution = np.random.normal(0,1, np.array(words).shape) #np.random.normal(np.array(words).shape)

            elif case == 'attn_last':
                attribution = np.mean([x_.sum(0) for x_ in model.attention_probs[max(layer_idxs)].detach().cpu().numpy()[0]],0)
            elif 'rollout' in case:
                attns = [model.attention_probs[k].detach().cpu().numpy() for k in sorted(model.attention_probs.keys())]
                attentions_mat = np.stack(attns,axis=0).squeeze()
                res_att_mat = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
                joint_attentions = compute_joint_attention(res_att_mat, add_residual=True)
                idx = int(case.replace('rollout_',''))
                attribution = joint_attentions[idx].sum(0)
            elif 'GAE' in case:
                attns = [model.attention_probs[k].detach().cpu().numpy() for k in sorted(model.attention_probs.keys())]
                attentions_mat = np.stack(attns,axis=0).squeeze()
                attns = [model.attention_gradients[k].detach().cpu().numpy() for k in sorted(model.attention_gradients.keys())]
                attentions_grads = np.stack(attns,axis=0).squeeze()
                attentions_mat = torch.tensor(attentions_mat * attentions_grads).clamp(min=0)
                attentions_mat = torch.tensor(attentions_mat).clamp(min=0).mean(dim=1)
                joint_attentions = _compute_rollout_attention(attentions_mat)
                joint_attentions[:, 0, 0] = 0
                idx = 0
                attribution = joint_attentions[idx].sum(0)
            elif 'attention_flow' in case:
                attns = [model.attention_probs[k].detach().cpu().numpy() for k in sorted(model.attention_probs.keys())]
                attentions_mat = np.stack(attns,axis=0).squeeze()
                idx = int(case.replace('attention_flow_',''))

                aflow = get_flow_relevance_for_all_layers(input_ids.detach().cpu().numpy().squeeze(),  
                                                          attentions_mat[:, np.newaxis],
                                                          tokens=words, 
                                                          layers=[idx],
                                                          pad_token= tokenizer.pad_token_id)
                assert len(aflow) == 1
                attribution =  aflow[0]
            elif case == 'naive_gi':
                outs = model.forward_and_explain(input_ids=input_ids, cl=y_true,
                               labels = labels_in,
                                 gammas = [0.0,0.0,0.0])

                attribution = outs['R'].squeeze()


            elif case in ['gi', 'gi_detach_KQ_LNorm', 'gi_detach_KQ', 'gi_detach_KQ_LNorm_Norm']:
                outs = model.forward_and_explain(input_ids=input_ids, cl=y_true,
                               labels = labels_in,
                               gammas = [0.0,0.0,0.0])

                attribution = outs['R'].squeeze()


            elif case in ['lrp', 'lrp_detach_KQ_LNorm', 'lrp_detach_KQ', 'lrp_detach_KQ_LNorm_Norm']:

                outs = model.forward_and_explain(input_ids=input_ids, cl=y_true,
                               labels = labels_in, 
                                gammas = gammas)


                attribution = outs['R'].squeeze()


            if j==0:
                print(type(attribution), type(input_ids),  type(words))

            m, e, evolution = flip(model,
                              x=attribution, 
                             token_ids=input_ids, 
                             tokens=words,
                             y_true=y_true, 
                             fracs=fracs, 
                             flip_case=flip_case,
                             random_order = random_order, 
                             tokenizer=tokenizer,
                             device=device)


            M.append(m)
            E.append(e)
            EVOLUTION.append(evolution)
            

            if j%500==0:
                print('****',j)


            j+=1
        all_flips[case]= {'E':E, 'M':M, 'Evolution':EVOLUTION} 

        res_conservation[case] = C

    f, axs = plt.subplots(1, 2, figsize=(14, 8))
    for k, v in all_flips.items():
        print(len(v['M']))
        axs[0].plot(np.nanmean(v['M'], axis=0), label=k)
        axs[0].set_title('($y_0$-$y_p$)$^2$')
        axs[1].plot(np.nanmean(v['E'], axis=0), label=k)
        axs[1].set_title('logits$_k$')    
    plt.legend()
    
    f.savefig('{}.png'.format(flip_case), dpi=300)
    pickle.dump(all_flips, open('all_flips_{}_sst.p'.format(flip_case), 'wb'))


