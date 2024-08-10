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
from utils.preprocess import preprocess_graph, make_data
from utils.sample import sample_negative_edges, sample_hard_negative_edges
from utils.metrics import metrics, recall_at_k
from utils.evaluation import evaluate_model
from utils.util import tensor2csv

def test(args, model, data, n_user, n_item, emb_path):
    
    print(n_user, "users, ", n_item, "items")
    batch_size = 64
    k = 10
  # based on recall
    with torch.no_grad():
        alpha = 1. / (args['num_layers'] + 1)
        alpha = torch.tensor([alpha] * (args['num_layers'] + 1))
        embeddings = torch.load(emb_path)
        weights = alpha.softmax(dim=-1)
        embeddings = embeddings * weights[0]
        #model.get_embedding(data.edge_index)
        user_embeddings = embeddings[:n_user]
        item_embeddings = embeddings[n_user:]

    # book data
    # have to change... ?
    #data.edge_index[1,:]=torch.clamp(data.edge_index[1,:], max=n_user+n_item-1, min = n_user)
    
    first = 0
    # choose users, size of batch size
    for batch_start in range(0, n_user, batch_size):
        batch_end = min(batch_start + batch_size, n_user)
        batch_user_embeddings = user_embeddings[batch_start:batch_end]

        # Calculate scores for all possible item pairs
        scores = torch.matmul(batch_user_embeddings, item_embeddings.t())

        # Set the scores of message passing edges to negative infinity
        mp_indices = ((data.edge_index[0] >= batch_start) & (data.edge_index[0] < batch_end)).nonzero(as_tuple=True)[0]
        scores[data.edge_index[0, mp_indices] - batch_start, data.edge_index[1, mp_indices]-n_user] = -float("inf")

        # Find the top k highest scoring items for each playlist in the batch
        _, top_k_indices = torch.topk(scores, k, dim=1)
        if(first): 
          recommend = torch.cat([recommend, top_k_indices])
        else:
          first = 1
          recommend = top_k_indices
    
    return recommend

def load_model(num_nodes, args, alpha = False):
    print("loading model...")
    model = GCN(
        num_nodes = num_nodes, num_layers = args['num_layers'], 
        embedding_dim = args["emb_size"], conv_layer = args['conv_layer'], 
        alpha_learnable = alpha
    )
    model.to(args["device"])
    #model.load(args["model path"])
    return model

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

    G, user_idx, item_idx, n_user, n_item, id2node = preprocess_graph(G)
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
        'emb path': '/content/BookGNN/model_embeddings/LGCN_LGC_4_e64_nodes17738_/LGCN_LGC_4_e64_nodes17738__BPR_random_20.pt'
    }

    test_split = torch.load('../datasets/test_split.pt')
    test_split.to(args['device'])

    #model = load_model(n_nodes, args)
    #model.to(args["device"])
    #model.eval()
    model = None

    recommend  = test(args, model, test_split, n_user, n_item, args['emb path'])
    TEST_RESULT_DIR = "test_result"
    if not os.path.exists(TEST_RESULT_DIR):
      os.makedirs(TEST_RESULT_DIR)

    torch.save(recommend, "test_result/recommend.pt")
    tensor2csv(recommend, "test_result/recommend.csv")
    print("test done.")
    #print("RECALL:", recall_at_k(test_split, None, n_user, n_item, k=10, 
    #                      device=args["device"]), emb_path=args['emb path'], num_layers=args['num_layers'])
    # add ftn to change id2node and display


    