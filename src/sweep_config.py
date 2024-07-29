import wandb

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'conv_layer': {
            'values': ['LGC', 'GAT', 'SAGE']
        },
        'epochs': {
            'values': [100, 200, 301]
        },
        'num_layers': {
            'values': [2, 3, 4]
        },
        'lr': {
            'min': 0.0001,
            'max': 0.1
        },
        'neg_samp': {
            'values': ['random', 'hard']
        },
        'emb_size': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [1e-5, 1e-4, 1e-3]
        },
        'gat_dropout': {
            'min': 0.1,
            'max': 0.5
        },
        'gat_n_heads': {
            'values': [1, 2, 4]
        }
    }
}


sweep_id = wandb.sweep(sweep_config, project="Prometheus-GNN-Book-Recommendations")
