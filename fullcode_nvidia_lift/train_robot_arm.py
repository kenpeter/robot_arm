#!/usr/bin/env python
"""
Simple Robot Arm RL Training Script

This script trains a Franka Panda robot arm to reach target positions using IsaacLab.
Run this with IsaacLab's launcher:
    cd /home/kenpeter/work/IsaacLab
    ./isaaclab.sh -p /home/kenpeter/work/robot_arm/train_robot_arm.py

You can modify the TASK variable below to train different robots/tasks.
"""

import argparse
import sys

# Configuration
TASK = "Isaac-Lift-Cube-Franka-IK-Rel-v0"  # Change this to train different tasks
# Other options:
#   "Isaac-Reach-Franka-v0"         - Franka reach only
#   "Isaac-Lift-Cube-Franka-v0"     - Franka lift cube (joint control)
#   "Isaac-Lift-Cube-Franka-IK-Abs-v0" - Franka lift cube (absolute IK)

# Launch Isaac Sim
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train robot arm with RL")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
# Note: --headless is added by AppLauncher, don't add it here
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch the app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything else
import gymnasium as gym
import os
import torch
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks

# Prevent TF32 issues
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    """Train the robot arm."""

    print("=" * 80)
    print(f"Training Task: {TASK}")
    print(f"Environments: {args.num_envs}")
    print(f"Iterations: {args.max_iterations}")
    print(f"Seed: {args.seed}")
    print(f"Headless: {args.headless}")
    print("=" * 80)

    # Create environment
    env_cfg = gym.make(TASK).unwrapped.cfg
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    # Setup logging
    log_root_path = os.path.join("logs", "rsl_rl", TASK)
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    print(f"Logging to: {log_dir}")

    # Create environment
    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Create agent config
    agent_cfg = RslRlOnPolicyRunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.experiment_name = TASK

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    # Save configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print("\nStarting training...")
    print(f"Monitor with: tensorboard --logdir {log_root_path}")
    print()

    # Train!
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {log_dir}")
    print(f"\nTo test the policy, run:")
    print(f"cd /home/kenpeter/work/IsaacLab")
    print(f"./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task {TASK}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
