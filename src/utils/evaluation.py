import os
import pickle
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score
from model.gcn import GCN
from utils.preprocess import preprocess_graph, make_data
from utils.sample import sample_negative_edges, sample_hard_negative_edges
from utils.metrics import metrics, recall_at_k

MODEL_STATS_DIR = "model_stats"
SAVE_DIR = "train_result_plots"

def load_stats(stats_file):
    stats_path = os.path.join(MODEL_STATS_DIR, stats_file)
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        return stats
    else:
        print(f"No stats found at {stats_path}")
        return None


def evaluate_model(stats_file, model_fil, args):
    stats = load_stats(stats_file)
    if not stats:
        return

    stats = load_stats(stats_file)
    if not stats:
        return

    parts = stats_file.replace('.pkl', '').rsplit('_', 3)
    if len(parts) < 4:
        print(f"Unexpected file name format: {stats_file}")
        return
    model_name = f"GCN_{args['conv_layer']}_layers{args['num_layers']}_e{args['emb_size']}_nodes{args['n_nodes']}"
    loss_fn = args['loss_fn']
    neg_samp = args['neg_samp']
    conv_layer = args['conv_layer']
    n_nodes = args['n_nodes']
    
    with open('../assets/graph_kcore.gpickle', 'rb') as f:
        G = pickle.load(f)
    
    G, user_idx, item_idx, n_user, n_item = preprocess_graph(G)
    n_nodes = G.number_of_nodes()
    _, val_split, test_split = make_data(G)
    
    model = GCN(
        num_nodes=n_nodes, num_layers=args['num_layers'],
        embedding_dim=args["emb_size"], conv_layer=args['conv_layer'], 
        alpha_learnable=False
    )
    model.to(args["device"])
    
    model_file = os.path.join(MODEL_STATS_DIR, f"{model.name}_{args['loss_fn']}_{args['neg_samp']}_final.pt")
    model.load_state_dict(torch.load(model_file, map_location=args["device"]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--stats_file', type=str, required=True, help='Name of the stats file')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the model file')
    args = parser.parse_args()
    
    plot_training_stats(args['stats_file'])
    evaluate_model(args['stats_file'], args['model_file'])
