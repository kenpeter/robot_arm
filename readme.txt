ROBOT ARM RL TRAINING - WORKING! ✅
====================================

✨ MODELS NOW SAVE IN robot_arm/logs/ ✨

Quick Start - Train Robot:
--------------------------
cd /home/kenpeter/work/robot_arm
./run_training.sh --num_envs 512 --max_iterations 1000 --headless

Resume Training (continue from last checkpoint):
------------------------------------------------
cd /home/kenpeter/work/robot_arm
./resume_training.sh --num_envs 512 --max_iterations 2000 --headless

This will automatically find and continue from your latest checkpoint!

Quick Start - Replay/Watch Trained Robot:
------------------------------------------
cd /home/kenpeter/work/robot_arm
./play_trained_policy.sh

This will:
- Load the latest trained checkpoint from robot_arm/logs/
- Show 32 Franka robots executing the learned policy
- Open GUI window so you can watch them!

List Your Saved Checkpoints:
----------------------------
./list_checkpoints.sh

This shows all your saved training runs and checkpoints!

Monitor Training Progress:
--------------------------
tensorboard --logdir logs/rsl_rl

Then open browser to: http://localhost:6006

Where Are Checkpoints Saved:
----------------------------
/home/kenpeter/work/robot_arm/logs/rsl_rl/franka_reach/<timestamp>/

Example:
/home/kenpeter/work/robot_arm/logs/rsl_rl/franka_reach/2025-11-22_14-50-45/
├── model_0.pt (initial)
├── model_100.pt
├── model_500.pt
└── model_999.pt (final)

You currently have:
- 2 training runs saved
- 23 checkpoints total
- Latest: model_999.pt (1000 iterations complete!)

Available Tasks:
---------------
Edit run_training.sh to change --task:
- Isaac-Reach-Franka-v0        (Franka reach with joint control) ← default
- Isaac-Reach-Franka-IK-Rel-v0 (Franka reach with IK)
- Isaac-Lift-Cube-Franka-v0    (Franka lift cube)
- Isaac-Reach-UR10-v0          (UR10 reach)

Training Parameters:
-------------------
./run_training.sh accepts these options:
  --num_envs 2048          Number of parallel robots (default: 4096)
  --max_iterations 1000    Training iterations
  --headless               No GUI (faster training)
  --seed 42                Random seed

Example - Quick Test:
--------------------
./run_training.sh --num_envs 64 --max_iterations 10 --headless

Example - Full Training:
------------------------
./run_training.sh --num_envs 2048 --max_iterations 3000 --headless

Files in This Directory:
-----------------------
run_training.sh        - Train robot (start new training)
resume_training.sh     - Resume training from latest checkpoint
play_trained_policy.sh - Replay/watch trained robot
list_checkpoints.sh    - List all saved checkpoints
logs/                  - All your training checkpoints saved here!
readme.txt             - This file
QUICK_REFERENCE.txt    - Quick commands

What Happens When You Train:
----------------------------
1. Script activates conda environment
2. Sets up Isaac Sim environment
3. Creates simulation with many parallel robots
4. Trains them using PPO reinforcement learning
5. Saves checkpoints to: robot_arm/logs/rsl_rl/

How It Works:
------------
- IsaacLab/logs is now a symlink to robot_arm/logs
- All checkpoints save directly in your project folder
- No need to copy files from IsaacLab anymore!

Troubleshooting:
---------------
If training fails, make sure:
1. You're in robot_arm directory
2. isaaclab_env conda environment exists
3. Run: conda activate isaaclab_env (should work without errors)

Manual Training (if script doesn't work):
------------------------------------------
conda activate isaaclab_env
source /home/kenpeter/work/IsaacLab/_isaac_sim/setup_conda_env.sh
cd /home/kenpeter/work/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-Franka-v0

That's it! Happy training! 🤖


./run_training --resume --headless