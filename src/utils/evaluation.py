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
SAVE_DIR = "saved_plots"

def load_stats(stats_file):
    stats_path = os.path.join(MODEL_STATS_DIR, stats_file)
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        return stats
    else:
        print(f"No stats found at {stats_path}")
        return None

def plot_training_stats(stats_file):
    stats = load_stats(stats_file)
    if not stats:
        return

    model_name, loss_fn, neg_samp = stats_file.replace('.pkl', '').split('_', 2)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    ax[0].set_title(f'{model_name} Training and Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    
    ax[1].set_title(f'{model_name} Training and Validation ROC')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('ROC AUC')
    
    epochs = range(len(stats['train']['loss']))
    train_loss = [loss.cpu().item() for loss in stats['train']['loss']]
    val_loss = [loss.cpu().item() for loss in stats['val']['loss']]
    train_roc = stats['train']['roc']
    val_roc = stats['val']['roc']
    
    ax[0].plot(epochs, train_loss, label='Train Loss')
    ax[0].plot(epochs, val_loss, label='Val Loss')
    ax[1].plot(epochs, train_roc, label='Train ROC')
    ax[1].plot(epochs, val_roc, label='Val ROC')
    
    ax[0].legend()
    ax[1].legend()
    
    plt.tight_layout()
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    save_path = f"{SAVE_DIR}/{model_name}_{loss_fn}_{neg_samp}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

def evaluate_model(stats_file, model_file):
    stats = load_stats(stats_file)
    if not stats:
        return

    model_name, loss_fn, neg_samp = stats_file.replace('.pkl', '').split('_', 2)
    
    # Load the graph data
    with open('../assets/graph_kcore.gpickle', 'rb') as f:
        G = pickle.load(f)
    
    G, user_idx, item_idx, n_user, n_item = preprocess_graph(G)
    n_nodes = G.number_of_nodes()
    _, val_split, test_split = make_data(G)
    
    args = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'emb_size': 64,
        'weight_decay': 1e-5,
        'lr': 0.01,
        'loss_fn': loss_fn,
        'epochs': 301,
        'num_layers': 4,
        'conv_layer': model_name.split('_')[1],
        'neg_samp': neg_samp,
    }
    
    # Initialize the model
    model = GCN(
        num_nodes=n_nodes, num_layers=args['num_layers'],
        embedding_dim=args["emb_size"], conv_layer=args['conv_layer'], 
        alpha_learnable=False
    )
    model.to(args["device"])

    # Load the trained model weights
    model.load_state_dict(torch.load(model_file, map_location=args["device"]))

    # Perform evaluation on the test data
    test_loss, test_roc, _, _ = test(model, test_split, args, n_user, n_item)
    print(f"Test Loss: {test_loss}, Test ROC: {test_roc}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--stats_file', type=str, required=True, help='Name of the stats file')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the model file')
    args = parser.parse_args()
    
    plot_training_stats(args.stats_file)
    evaluate_model(args.stats_file, args.model_file)
