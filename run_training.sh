#!/bin/bash

# ==============================================================================
# Unified IsaacLab RL Training Script
# 
# Usage:
#   Start new training: ./run_rl.sh
#   Resume latest run:  ./run_rl.sh --resume
#   Pass extra args:    ./run_rl.sh --headless --max_iterations 1000
# ==============================================================================

ISAACLAB_DIR="/home/kenpeter/work/IsaacLab"
WORK_DIR="/home/kenpeter/work/robot_arm"
LOGS_DIR="$WORK_DIR/logs"
TASK_NAME="Isaac-Lift-Cube-Franka-IK-Rel-v0"
# Note: The folder name below usually matches the task name but processed (e.g. franka_lift)
# Update this if you change tasks frequently
REL_LOG_PATH="rsl_rl/franka_lift" 

echo "=========================================="
echo "      Robot Arm RL Training Manager       "
echo "=========================================="

# ------------------------------------------------------------------------------
# 1. Environment Setup
# ------------------------------------------------------------------------------

echo ">> Activating conda environment (isaaclab_env)..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate isaaclab_env

echo ">> Setting up Isaac Sim environment..."
source "$ISAACLAB_DIR/_isaac_sim/setup_conda_env.sh"

# ------------------------------------------------------------------------------
# 2. Log Directory & Symlink Setup
# ------------------------------------------------------------------------------

# Create logs directory in robot_arm if it doesn't exist
mkdir -p "$LOGS_DIR"

# Create symlink so models save to robot_arm/logs instead of IsaacLab/logs
if [ ! -L "$ISAACLAB_DIR/logs" ]; then
    echo ">> Setting up logs symlink..."
    # Backup existing logs if any
    if [ -d "$ISAACLAB_DIR/logs" ]; then
        echo "   Backing up existing IsaacLab logs..."
        mv "$ISAACLAB_DIR/logs" "$ISAACLAB_DIR/logs_backup_$(date +%Y%m%d_%H%M%S)"
    fi
    # Create symlink
    ln -s "$LOGS_DIR" "$ISAACLAB_DIR/logs"
    echo "   ✓ Models will now save to: $LOGS_DIR"
fi

# ------------------------------------------------------------------------------
# 3. Argument Parsing & Resume Logic
# ------------------------------------------------------------------------------

# Change to IsaacLab directory
cd "$ISAACLAB_DIR" || exit

ARGS=("$@")
IS_RESUMING=false
HAS_LOAD_RUN=false

# Check arguments for --resume and --load_run
for arg in "${ARGS[@]}"; do
    if [ "$arg" == "--resume" ]; then
        IS_RESUMING=true
    fi
    if [ "$arg" == "--load_run" ]; then
        HAS_LOAD_RUN=true
    fi
done

# Logic: If resuming but user didn't specify WHICH run, find the latest one
if [ "$IS_RESUMING" = true ] && [ "$HAS_LOAD_RUN" = false ]; then
    echo ">> Resume flag detected. searching for latest checkpoint..."
    
    # Find the latest run directory
    # Note: We look inside the robot_arm logs folder directly
    FULL_SEARCH_PATH="$LOGS_DIR/$REL_LOG_PATH/*/"
    LATEST_RUN=$(ls -td $FULL_SEARCH_PATH 2>/dev/null | head -1)

    if [ -z "$LATEST_RUN" ]; then
        echo "❌ Error: No previous training runs found in $FULL_SEARCH_PATH"
        echo "   Cannot resume. Starting fresh or check your paths."
        exit 1
    fi

    LATEST_RUN_NAME=$(basename "$LATEST_RUN")
    
    echo "   ✓ Found latest run: $LATEST_RUN_NAME"
    echo "   ✓ Checkpoints loading from: $LATEST_RUN"
    
    # Add the load_run argument to the list
    ARGS+=("--load_run" "$LATEST_RUN_NAME")
else
    if [ "$IS_RESUMING" = true ]; then
        echo ">> Resuming specific run defined in arguments..."
    else
        echo ">> Starting NEW training session..."
        echo "   Checkpoints will be saved to: $LOGS_DIR/rsl_rl/"
    fi
fi

# ------------------------------------------------------------------------------
# 4. Execution
# ------------------------------------------------------------------------------

echo ""
echo ">> Running IsaacLab..."
echo "----------------------------------------------------------------"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK_NAME" \
    "${ARGS[@]}"

echo ""
echo "=========================================="
echo "Training script finished."
echo "=========================================="