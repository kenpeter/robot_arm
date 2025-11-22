#!/bin/bash

# Script to replay/test a trained robot arm policy
# This will visualize the trained robot in action

ISAACLAB_DIR="/home/kenpeter/work/IsaacLab"

echo "=========================================="
echo "Playing Trained Robot Arm Policy"
echo "=========================================="

# Activate conda environment
echo "Activating isaaclab_env..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate isaaclab_env

# Source Isaac Sim environment setup
echo "Setting up Isaac Sim environment..."
source "$ISAACLAB_DIR/_isaac_sim/setup_conda_env.sh"

# Change to IsaacLab directory
cd "$ISAACLAB_DIR"

# Run the play script with GUI (no --headless)
echo "Loading trained policy and visualizing..."
echo ""
echo "This will open a window showing the robots in action!"
echo ""

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 32 \
    "$@"
