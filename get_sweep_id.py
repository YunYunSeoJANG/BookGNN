import wandb
import yaml

# Set the project name
project_name = 'Prometheus-GNN-Book-Recommendations'

# Load the YAML file
with open('src/sweep_config.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)

# Create the sweep from the dictionary
sweep_id = wandb.sweep(sweep_config, project=project_name)

# Print only the sweep ID
print(sweep_id)