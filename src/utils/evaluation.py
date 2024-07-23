import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from argparse import ArgumentParser

MODEL_STATS_DIR = "model_stats"
MODEL_EMBEDDINGS_DIR = "model_embeddings"
os.makedirs(MODEL_STATS_DIR, exist_ok=True)
os.makedirs(MODEL_EMBEDDINGS_DIR, exist_ok=True)

def load_stats(model_name, loss_fn, neg_samp):
    stats_path = f"{MODEL_STATS_DIR}/{model_name}_{loss_fn}_{neg_samp}.pkl"
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        return stats
    else:
        print(f"No stats found at {stats_path}")
        return None

def initialize_plot(model_name):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].set_title(f'{model_name} Training and Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].set_title(f'{model_name} Training and Validation ROC')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('ROC AUC')
    train_loss_line, = ax[0].plot([], [], label='Train Loss')
    val_loss_line, = ax[0].plot([], [], label='Val Loss')
    train_roc_line, = ax[1].plot([], [], label='Train ROC')
    val_roc_line, = ax[1].plot([], [], label='Val ROC')
    ax[0].legend()
    ax[1].legend()
    return fig, ax, train_loss_line, val_loss_line, train_roc_line, val_roc_line

def update_plot(stats, lines, ax):
    epochs = range(len(stats['train']['loss']))
    lines[0].set_data(epochs, stats['train']['loss'])
    lines[1].set_data(epochs, stats['val']['loss'])
    lines[2].set_data(epochs, stats['train']['roc'])
    lines[3].set_data(epochs, stats['val']['roc'])
    ax[0].relim()
    ax[0].autoscale_view()
    ax[1].relim()
    ax[1].autoscale_view()
    plt.draw()
    plt.pause(0.001)

def plot_training_stats(model_name, loss_fn, neg_samp):
    stats = load_stats(model_name, loss_fn, neg_samp)
    if not stats:
        return
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    ax[0].set_title(f'{model_name} Training and Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    
    ax[1].set_title(f'{model_name} Training and Validation ROC')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('ROC AUC')
    
    train_loss_line, = ax[0].plot([], [], label='Train Loss')
    val_loss_line, = ax[0].plot([], [], label='Val Loss')
    train_roc_line, = ax[1].plot([], [], label='Train ROC')
    val_roc_line, = ax[1].plot([], [], label='Val ROC')
    
    ax[0].legend()
    ax[1].legend()
    
    lines = [train_loss_line, val_loss_line, train_roc_line, val_roc_line]
    
    ani = FuncAnimation(
        fig, update_plot, frames=range(len(stats['train']['loss'])), fargs=(stats, lines, ax),
        repeat=False, blit=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--loss_fn', type=str, required=True, help='Loss function used')
    parser.add_argument('--neg_samp', type=str, required=True, help='Negative sampling method used')
    args = parser.parse_args()
    
    plot_training_stats(args.model_name, args.loss_fn, args.neg_samp)
