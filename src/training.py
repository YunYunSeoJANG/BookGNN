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
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from model.gcn import GCN, BPRLoss
from utils.preprocess import preprocess_graph, make_data
from utils.sample import sample_negative_edges, sample_hard_negative_edges
from utils.metrics import metrics, recall_at_k


def load_graph():
    # Read interactions from preprocessed json file in data_preprocessing.ipynb
    with open('../datasets/interactions_poetry.json') as f:
      users = pd.DataFrame(json.loads(line) for line in f)

    # is it necessary? it doesn't seem to be used
    with open('../datasets/books_poetry.json') as f:
      items = pd.DataFrame(json.loads(line) for line in f)

    # Make an empty graph
    G = nx.Graph()

    # Add user nodes to the Graph (377799 users)
    G.add_nodes_from(users['user_id'], type = 'user')
    # print('Num nodes:', G.number_of_nodes(), '. Num edges:', G.number_of_edges()) # Num nodes: 377799 . Num edges: 0

    # Add item nodes to the graph (36514 books)
    G.add_nodes_from(users['book_id'], type = 'book')

    # Make a bipartite graph
    edges = [(row['user_id'], row['book_id']) for index, row in users.iterrows()]
    G.add_edges_from(edges)
    
    # print('Num nodes:', G.number_of_nodes(), '. Num edges:', G.number_of_edges()) # Num nodes: 414313 . Num edges: 2734350

    kcore = 30
    G = nx.k_core(G, kcore)
    # print('Num nodes:', G.number_of_nodes(), '. Num edges:', G.number_of_edges()) # Num nodes: 17738 . Num edges: 767616

    # print(list(G.nodes(data=True))[:5])


    os.makedirs('../assets', exist_ok=True)
    with open('../assets/graph_kcore.gpickle', 'wb') as f:
        pickle.dump(G, f)

    return G


# Train
def train(model, datasets, optimizer, args, n_user, n_item):
  print(f"Beginning training for {model.name}")

  train_data = datasets["train"]
  val_data = datasets["val"]

  stats = {
      'train': {
        'loss': [],
        'roc' : []
      },
      'val': {
        'loss': [],
        'recall': [], 
        'roc' : []
      }

  }
  val_neg_edge, val_neg_label = None, None
  for epoch in range(args["epochs"]): # loop over each epoch 
    model.train()
    optimizer.zero_grad()
    
    # obtain negative sample
    if args['neg_samp'] == "random":
      neg_edge_index, neg_edge_label = sample_negative_edges(train_data, n_user, n_item, args["device"])
    elif args['neg_samp'] == "hard":
      if epoch % 5 == 0: 
        neg_edge_index, neg_edge_label = sample_hard_negative_edges(
            train_data, model, n_user, n_item, args["device"], batch_size = 500, 
            frac_sample = 1 - (0.5 * epoch / args["epochs"])
        )
    # calculate embedding
    embed = model.get_embedding(train_data.edge_index)
    # calculate pos, negative scores using embedding
    pos_scores = model.predict_link_embedding(embed, train_data.edge_label_index)
    neg_scores = model.predict_link_embedding(embed, neg_edge_index)

    # concatenate pos, neg scores together and evaluate loss 
    scores = torch.cat((pos_scores, neg_scores), dim = 0)
    labels = torch.cat((train_data.edge_label, neg_edge_label), dim = 0)

    # calculate loss function 
    if args['loss_fn'] == "BCE": 
      loss = model.link_pred_loss(scores, labels)
    elif args['loss_fn'] == "BPR":
      loss = model.recommendation_loss(pos_scores, neg_scores, lambda_reg = 0)
    
    train_roc = metrics(labels, scores)

    loss.backward()
    optimizer.step()

    val_loss, val_roc, val_neg_edge, val_neg_label = test(
        model, val_data, args, n_user, n_item, epoch, val_neg_edge, val_neg_label
    )

    stats['train']['loss'].append(loss)
    stats['train']['roc'].append(train_roc)
    stats['val']['loss'].append(val_loss)
    stats['val']['roc'].append(val_roc)

    print(f"Epoch {epoch}; Train loss {loss}; Val loss {val_loss}; Train ROC {train_roc}; Val ROC {val_roc}")

    if epoch % 10 == 0: 
      # calculate recall @ K
      # Suggestion: K -> 15~30
      val_recall = recall_at_k(val_data, model, n_user, n_item, k = 300, device = args["device"])
      print(f"Val recall {val_recall}")
      stats['val']['recall'].append(val_recall)

    if epoch % 20 == 0: 

      # save embeddings for future visualization 
      path = os.path.join("model_embeddings", model.name)
      if not os.path.exists(path):
        os.makedirs(path)
      torch.save(model.embedding.weight, os.path.join("model_embeddings", model.name, f"{model.name}_{args['loss_fn']}_{args['neg_samp']}_{epoch}.pt"))

  pickle.dump(stats, open(f"model_stats/{model.name}_{args['loss_fn']}_{args['neg_samp']}.pkl", "wb"))
  return stats


def test(model, data, args, n_user, n_item, epoch = 0, neg_edge_index = None, neg_edge_label = None):
  
  model.eval()
  with torch.no_grad(): # want to save RAM 

    # conduct negative sampling 
    if args['neg_samp'] == "random":
      neg_edge_index, neg_edge_label = sample_negative_edges(data, n_user, n_item, args["device"])
    elif args['neg_samp'] == "hard":
      if epoch % 5 == 0 or neg_edge_index is None: 
        neg_edge_index, neg_edge_label = sample_hard_negative_edges(
            data, model, n_user, n_item, args["device"], batch_size = 500,
            frac_sample = 1 - (0.5 * epoch / args["epochs"])
        )
    # obtain model embedding
    embed = model.get_embedding(data.edge_index)
    # calculate pos, neg scores using embedding 
    pos_scores = model.predict_link_embedding(embed, data.edge_label_index)
    neg_scores = model.predict_link_embedding(embed, neg_edge_index)
    # concatenate pos, neg scores together and evaluate loss 
    scores = torch.cat((pos_scores, neg_scores), dim = 0)
    labels = torch.cat((data.edge_label, neg_edge_label), dim = 0)
    # calculate loss 
    if args['loss_fn'] == "BCE": 
      loss = model.link_pred_loss(scores, labels)
    elif args['loss_fn'] == "BPR":
      loss = model.recommendation_loss(pos_scores, neg_scores, lambda_reg = 0)

    roc = metrics(labels, scores)
    
  return loss, roc, neg_edge_index, neg_edge_label

def init_model(num_nodes, args, alpha = False):
    model = GCN(
        num_nodes = num_nodes, num_layers = args['num_layers'], 
        embedding_dim = args["emb_size"], conv_layer = args['conv_layer'], 
        alpha_learnable = alpha
    )
    model.to(args["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    return model, optimizer 

if __name__ == '__main__':
    is_load_graph = not os.path.exists('assets/graph_kcore.gpickle')

    if is_load_graph:
        print("First run. Loading graph from json file and saving it as a gpickle file.")
        G = load_graph()
    else:
        print("Loading graph from gpickle file.")
        with open('assets/graph_kcore.gpickle', 'rb') as f:
           G = pickle.load(f)

    G, user_idx, item_idx, n_user, n_item = preprocess_graph(G)
    n_nodes = G.number_of_nodes()
    train_split, val_split, test_split = make_data(G)

    # create a dictionary of the dataset splits 
    datasets = {
        'train':train_split, 
        'val':val_split, 
        'test': test_split 
    }

    # Modify the arguments as needed
    # You might want to change the epoches, num_layers, conv_layer, neg_samp, etc.
    args = {
        'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
        'emb_size' : 64,
        'weight_decay': 1e-5,
        'lr': 0.01,
        'loss_fn': "BPR",
        'epochs': 301, # [301, 150]
        'num_layers' :  3, # [3, 4]
        'conv_layer': "LGC", # ["LGC", "GAT", "SAGE"]
        'neg_samp': "random", # ["random", "hard"]
    }

    model, optimizer = init_model(n_nodes, args)

    # send data, model to GPU if available
    user_idx = torch.Tensor(user_idx).type(torch.int64).to(args["device"])
    item_idx =torch.Tensor(item_idx).type(torch.int64).to(args["device"])
    datasets['train'].to(args['device'])
    datasets['val'].to(args['device'])
    datasets['test'].to(args['device'])
    model.to(args["device"])
    
    # create directory to save model_stats
    MODEL_STATS_DIR = "model_stats"
    if not os.path.exists(MODEL_STATS_DIR):
      os.makedirs(MODEL_STATS_DIR)


    train(model, datasets, optimizer, args, n_user, n_item)

    test(model, datasets['test'], args, n_user, n_item)


    