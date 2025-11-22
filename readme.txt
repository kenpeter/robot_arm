# Pick and Place Robot Arm Training with IsaacLab

## Overview
This setup trains a **Franka Panda robot arm** to perform **pick and place** tasks using **reinforcement learning (RL)**.

## Current Task: `Isaac-Lift-Cube-Franka-IK-Rel-v0`

### What This Task Does
The robot learns to:
1. **Reach** towards a cube on a table
2. **Grasp** the cube using gripper control
3. **Lift** the cube off the table
4. **Move** it to a target position
5. **Place** it at the goal location

### Why IK Relative Control?
- **IK (Inverse Kinematics)**: Robot controls end-effector position/orientation instead of joint angles
- **Relative Control**: Commands are relative position changes (easier for RL to learn)
- **Better for manipulation**: More intuitive than joint-level control for pick/place

## Training Algorithm
- **Algorithm**: PPO (Proximal Policy Optimization) - standard RL algorithm
- **Library**: RSL-RL (Robotic Systems Lab RL library)
- **Type**: Model-free on-policy reinforcement learning

## Exact Code Location
All task code copied from IsaacLab to:
```
/home/kenpeter/work/robot_arm/lift_task/
├── config/franka/
│   ├── __init__.py              # Task registration
│   ├── joint_pos_env_cfg.py     # Joint position control config
│   ├── ik_rel_env_cfg.py        # IK relative control config (ACTIVE)
│   └── ik_abs_env_cfg.py        # IK absolute control config
├── lift_env_cfg.py               # Base lift environment configuration
└── mdp/                          # MDP (rewards, observations, terminations)
    ├── observations.py
    ├── rewards.py
    └── terminations.py
```

## Key Configuration Details

### Robot Configuration
- **Robot**: Franka Panda (7-DOF arm + 2-finger gripper)
- **Config**: `FRANKA_PANDA_HIGH_PD_CFG` (stiff PD controller for IK)
- **Source**: `isaaclab_assets.robots.franka`

### Actions
```python
# Arm: IK control (end-effector position/orientation deltas)
arm_action = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
    controller=DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls"  # Damped Least Squares IK solver
    ),
    scale=0.5,
    body_offset=[0.0, 0.0, 0.107]  # Gripper offset
)

# Gripper: Binary open/close
gripper_action = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04},   # 4cm open
    close_command_expr={"panda_finger_.*": 0.0}    # Closed
)
```

### Rewards (from lift_env_cfg.py)
1. **reaching_object**: Distance between gripper and cube (weight: 1.0)
2. **lifting_object**: Bonus for lifting cube above 4cm (weight: 15.0)
3. **object_goal_tracking**: Distance from cube to goal position (weight: 16.0)
4. **object_goal_tracking_fine_grained**: Fine-grained goal tracking (weight: 5.0)
5. **action_rate**: Penalty for large actions (weight: -1e-4)
6. **joint_vel**: Penalty for high joint velocities (weight: -1e-4)

### Observations
- Joint positions (relative)
- Joint velocities (relative)
- Object position (in robot frame)
- Target object position (commanded pose)
- Previous actions

### Object
- **Type**: Cube (80% scale of standard dex cube)
- **Initial Position**: [0.5, 0, 0.055] (on table)
- **Physics**: Rigid body with realistic friction/damping

## Usage

### Train the Robot
```bash
cd /home/kenpeter/work/robot_arm

# Quick test (512 environments, 1000 iterations)
./run_training.sh --num_envs 512 --max_iterations 1000 --headless

# Full training (recommended)
./run_training.sh --num_envs 4096 --max_iterations 10000 --headless
```

### Watch Trained Robot
```bash
./play_trained_policy.sh

# Or with specific checkpoint
./play_trained_policy.sh --checkpoint /path/to/model.pt
```

### Resume Training
```bash
# Automatically finds latest checkpoint
./run_training.sh --num_envs 4096 --max_iterations 15000 --resume
```

## Available Alternative Tasks

You can switch tasks by editing `run_training.sh` and `play_trained_policy.sh`:

| Task ID | Description | Control Type |
|---------|-------------|--------------|
| `Isaac-Lift-Cube-Franka-IK-Rel-v0` | **Pick & place cube** (relative IK) ← **ACTIVE** | IK Relative |
| `Isaac-Lift-Cube-Franka-IK-Abs-v0` | Pick & place cube (absolute IK) | IK Absolute |
| `Isaac-Lift-Cube-Franka-v0` | Pick & place cube (joint control) | Joint Position |
| `Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0` | Pick & place deformable teddy bear | IK Absolute |
| `Isaac-Reach-Franka-v0` | Simple reach (no grasp/lift) | Joint Position |

## Training Expectations

### Episode Details
- **Episode Length**: 5 seconds (500 steps at 100Hz with decimation=2)
- **Number of Environments**: 4096 parallel robots (configurable)
- **Termination**: Episode ends if cube drops below table

### Training Time
- **Quick test** (1000 iterations): ~5-10 minutes
- **Good policy** (5000 iterations): ~30-45 minutes
- **Publication-quality** (10000+ iterations): 1-2 hours

*Times vary based on GPU performance*

### Expected Learning Curve
1. **0-1000 iterations**: Robot learns to reach towards cube
2. **1000-3000 iterations**: Learns to grasp cube
3. **3000-5000 iterations**: Learns to lift cube
4. **5000-8000 iterations**: Learns to move to goal
5. **8000+ iterations**: Fine-tunes placement accuracy

## Logs and Checkpoints
All training data saved to:
```
/home/kenpeter/work/robot_arm/logs/rsl_rl/franka_cube_lift_ik_rel/
├── YYYY-MM-DD_HH-MM-SS/
│   ├── model_1000.pt
│   ├── model_2000.pt
│   ├── ...
│   └── params/
│       └── env.pkl
```

View checkpoints:
```bash
./list_checkpoints.sh
```

## Source Code Reference
- **Original IsaacLab code**: `/home/kenpeter/work/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/`
- **Task registration**: `config/franka/__init__.py:69-77`
- **IK config**: `config/franka/ik_rel_env_cfg.py:19-36`
- **Base config**: `lift_env_cfg.py:194-223`

## Key Differences from Reach Task

| Aspect | Reach Task | Lift Task (Pick & Place) |
|--------|------------|--------------------------|
| **Goal** | Move end-effector to target | Grasp object and move to target |
| **Actions** | Arm only | Arm + Gripper |
| **Observations** | End-effector pose | End-effector + Object pose |
| **Rewards** | Distance to target | Reach + Grasp + Lift + Place |
| **Complexity** | Simple | Complex (contact dynamics) |
| **Training Time** | Fast (~5 min) | Moderate (~30-60 min) |

## Notes
- This is **exact code from IsaacLab** - no custom modifications
- Training runs entirely within IsaacLab's framework
- Uses GPU acceleration (requires NVIDIA GPU)
- All scripts automatically activate conda environment and Isaac Sim setup




./run_training.sh --num_envs 4096 --max_iterations 5000 --headless


./run_training.sh --num_envs 4096 --max_iterations 10000 --resume --headless