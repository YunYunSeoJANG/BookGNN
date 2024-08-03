import json, pandas as pd
import networkx as nx
from typing import Optional, Union
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.nn import Embedding, ModuleList, Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn.conv import LGConv, GATConv, SAGEConv
from torch_geometric.typing import Adj, OptTensor, SparseTensor
import os 
from os.path import isfile, join
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from model.gcn import GCN, BPRLoss
from utils.preprocess import preprocess_graph, make_data, make_data_for_test
from utils.sample import sample_negative_edges, sample_hard_negative_edges
from utils.metrics import metrics, recall_at_k
from utils.evaluation import evaluate_model, plot_training_stats

def test(model, data, args, n_user, n_item):

  model.eval()
  with torch.no_grad(): # want to save RAM 

    # conduct negative sampling 
    if args['neg_samp'] == "random":
      neg_edge_index, neg_edge_label = sample_negative_edges(data, n_user, n_item, args["device"])
    elif args['neg_samp'] == "hard":
      neg_edge_index, neg_edge_label = sample_hard_negative_edges(
            data, model, n_user, n_item, args["device"], batch_size = 500,
            frac_sample = 1 - (0.5 * 100 / args["epochs"])
            )
    # obtain model embedding
    embed = model.get_embedding(data.edge_index)
    # calculate pos, neg scores using embedding 
    pos_scores = model.predict_link_embedding(embed, data.edge_label_index)
    neg_scores = model.predict_link_embedding(embed, neg_edge_index)
    # concatenate pos, neg scores together and evaluate loss 
    scores = torch.cat((pos_scores, neg_scores), dim = 0)
    labels = torch.cat((data.edge_label, neg_edge_label), dim = 0)
    
  return scores, labels, neg_edge_index, neg_edge_label

def load_model(num_nodes, args, alpha = False):
    print("loading model...")
    model = GCN(
        num_nodes = num_nodes, num_layers = args['num_layers'], 
        embedding_dim = args["emb_size"], conv_layer = args['conv_layer'], 
        alpha_learnable = alpha
    )
    model.to(args["device"])
    model.load(args["model path"])
    return model

def add_test_users(G):
  with open('../assets/node2id.pkl', 'rb') as file:
    node2id = pickle.load(file)
  
  with open('../datasets/test_user1.json') as f:
    users = pd.DataFrame(json.loads(line) for line in f)

  n_nodes = G.number_of_nodes()
  users['user_id'] = n_nodes
  G.add_nodes_from(users['user_id'], type = 'user')

  edges = [(row['user_id'], node2id[row['book_id']]) for index, row in users.iterrows()]
  G.add_edges_from(edges)
  return G
   
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--neg_samp', type=str, default="random", choices=["random", "hard"]) # ["random", "hard"]
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--conv_layer', type=str, default="LGC", choices=["LGC", "GAT", "SAGE"]) # ["LGC", "GAT", "SAGE"]
   
    parse_args = parser.parse_args()


    is_load_graph = not os.path.exists('../assets/graph_kcore.gpickle')

    if is_load_graph:
      print("No gpickle file.")
    
    with open('../assets/graph_kcore.gpickle', 'rb') as f:
      G = pickle.load(f)
    G = add_test_users(G)

    G, user_idx, item_idx, n_user, n_item = preprocess_graph(G)
    n_nodes = G.number_of_nodes()

    # Modify the arguments as needed
    # You might want to change the epoches, num_layers, conv_layer, neg_samp, etc.
    args = {
        'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
        'emb_size' : 64,
        'num_layers' : parse_args.num_layers, # [3, 4]
        'conv_layer': parse_args.conv_layer, # ["LGC", "GAT", "SAGE"]
        'neg_samp': parse_args.neg_samp, # ["random", "hard"]
        'n_nodes': n_nodes, # [17738]
        'model path': "..."
    }

    model = load_model(n_nodes, args)

    # send data, model to GPU if available
    user_idx = torch.Tensor(user_idx).type(torch.int64).to(args["device"])
    item_idx =torch.Tensor(item_idx).type(torch.int64).to(args["device"])
    model.to(args["device"])
    model.eval()

    train_split, val_split, test_split = make_data_for_test(G)
    datasets = {
        'train':train_split,
        'test': test_split 
    }

    user_idx = torch.Tensor(user_idx).type(torch.int64).to(args["device"])
    item_idx =torch.Tensor(item_idx).type(torch.int64).to(args["device"])
    datasets['train'].to(args['device'])
    datasets['test'].to(args['device'])
    model.to(args["device"])

    test(model, datasets, args, n_user, n_item)

    