from transformers import *
import numpy as np
from tqdm import tqdm

import math
import pandas as pd

import networkx as nx
import os
import torch

def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 


def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index[str(k)+"_"+input_tokens[k]] = k

    for i in np.arange(1,n_layers+1):
        for k_f in np.arange(length):
            index_from = (i)*length+k_f
            label = "L"+str(i)+"_"+str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i-1)*length+k_t
                adj_mat[index_from][index_to] = mat[i-1][k_f][k_t]
                
    return adj_mat, labels_to_index 


    
def compute_flows(G, labels_to_index, input_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in labels_to_index:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()
            
    return flow_values

def compute_node_flow(G, labels_to_index, input_nodes, output_nodes,length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in output_nodes:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()
            
    return flow_values


def compute_joint_attention(att_mat, add_residual=True):

    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None, ...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat
    
    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])
        
    return joint_attentions


def get_raw_att_relevance(full_att_mat, input_tokens, layer=-1):
    cls_index = 0
    return full_att_mat[layer].sum(axis=0).sum(axis=0)
    

def compute_node_flow(G, labels_to_index, input_nodes, output_nodes,length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in output_nodes:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v)
                flow_values[u][pre_layer*length+v] = flow_value
            #normalize flow values
            flow_values[u] /= flow_values[u].sum()
            
    return flow_values

def get_flow_relevance(full_att_mat, input_tokens, layer):
    '''
    Quantifying Attention Flow in Transformers (S Abnar and W Zuidema, ACL 2020)
    https://github.com/samiraabnar/attention_flow
    '''
    
    res_att_mat = full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None,...]
    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]

    res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=input_tokens)
    
    A = res_adj_mat
    res_G=nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(res_G, {(i,j): A[i,j]}, 'capacity')

    output_nodes = []
    input_nodes = []
    for key in res_labels_to_index:
        if key.startswith('L'+str(layer+1)+'_'):
            output_nodes.append(key)
        if res_labels_to_index[key] < full_att_mat.shape[-1]:
            input_nodes.append(key)
    
    flow_values = compute_node_flow(res_G, res_labels_to_index, input_nodes, output_nodes, length=full_att_mat.shape[-1])
    
    n_layers = full_att_mat.shape[0]
    length = full_att_mat.shape[-1]
    final_layer_attention_raw = flow_values[(layer+1)*length: (layer+2)*length,layer*length: (layer+1)*length]
    relevance_attention_raw = final_layer_attention_raw.sum(axis=0)

    return relevance_attention_raw
    
    
def get_joint_relevance(full_att_mat, input_tokens, layer):
    att_sum_heads =  full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    joint_attentions = compute_joint_attention(att_sum_heads, add_residual=True)
    relevance_attentions = joint_attentions[layer].sum(axis=0)
    return relevance_attentions


def get_nores_joint_relevance(full_att_mat, input_tokens, layer):
    att_sum_heads =  full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    joint_attentions = compute_joint_attention(att_sum_heads, add_residual=False)
    relevance_attentions = joint_attentions[layer].sum(axis=0)
    return relevance_attentions


def get_flow_relevance_for_all_layers(encoded, x, tokens, layers, pad_token):      
    is_token = np.array(encoded) != pad_token
    sen_len = x.shape[-1]
    assert  len(is_token) == sen_len == x.shape[-1]
    valid_token_len = is_token.sum()
    assert x.shape[1]==1
    attn_cropped = x[:,0,:,:valid_token_len, :valid_token_len]
    tokens_cropped = tokens[:valid_token_len]
    all_layers_flow_relevance=[]
    for l in layers:
        attention_relevance = get_flow_relevance(attn_cropped, tokens_cropped, layer=l)
        attention_relevance = np.concatenate((attention_relevance, np.array([-1.]*(sen_len-valid_token_len))), axis=None)
        all_layers_flow_relevance.append(attention_relevance)
        
    return all_layers_flow_relevance


def compute_joint_attention(att_mat, add_residual=True):
    '''
    Quantifying Attention Flow in Transformers (S Abnar and W Zuidema, ACL 2020)
    https://github.com/samiraabnar/attention_flow
    '''
    
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None,...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
    else:
        aug_att_mat =  att_mat
    
    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1,layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])
        
    return joint_attentions


def _compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention