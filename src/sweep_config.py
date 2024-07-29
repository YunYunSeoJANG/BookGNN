import wandb

sweep_config = {
    'method': 'grid',  # 'random', 'bayes'
    'metric': {
        'name': 'val_roc',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [150, 301]
        },
        'num_layers': {
            'values': [2, 3, 4, 5]
        },
        'lr': {
            'values': [0.01, 0.001, 0.0001]
        },
        'conv_layer': {
            'values': ["LGC", "GAT", "SAGE"]
        },
        'neg_samp': {
            'values': ["random", "hard"]
        },
        'emb_size': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [1e-4, 1e-5, 1e-6]
        },
        'gat_dropout': {
            'values': [0.2, 0.3, 0.4, 0.5]
        },
        'gat_n_heads': {
            'values': [4, 5, 6, 7]
        }
    }
}


sweep_id = wandb.sweep(sweep_config, project="Prometheus-GNN-Book-Recommendations")
