#!/bin/bash

# Script exits immediately if a command exits with a non-zero status
set -e

# Use the get_sweep_id.py script to generate the sweep and get the ID
SWEEP_ID=$(python3 get_sweep_id.py | grep -Eo '^[a-zA-Z0-9]+$')

echo "Sweep ID: $SWEEP_ID"

# Run the wandb agent with the project and entity
wandb agent --project Prometheus-GNN-Book-Recommendations --entity pljh0906 $SWEEP_ID &
wait
