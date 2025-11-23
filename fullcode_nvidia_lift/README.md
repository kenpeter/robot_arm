# NVIDIA Isaac-Lift-Cube-Franka-IK-Rel-v0 - Complete Implementation

This directory contains the **complete, self-contained implementation** of NVIDIA's `Isaac-Lift-Cube-Franka-IK-Rel-v0` task from IsaacLab, **including all RL training code**. This is a pick-and-place task where a Franka Panda robot learns to pick up a cube and place it at target locations using PPO.

## Directory Structure

```
fullcode_nvidia_lift/
├── lift/                           # Task Implementation (Environment)
│   ├── lift_env_cfg.py            # Base environment configuration
│   ├── __init__.py                # Task registration
│   ├── config/                    # Robot-specific configs
│   │   └── franka/
│   │       ├── joint_pos_env_cfg.py    # Joint position control
│   │       ├── ik_rel_env_cfg.py       # IK relative control (ACTIVE)
│   │       ├── ik_abs_env_cfg.py       # IK absolute control
│   │       └── agents/
│   │           └── rsl_rl_ppo_cfg.py   # PPO hyperparameters
│   └── mdp/                       # MDP components
│       ├── rewards.py             # Reward functions
│       ├── observations.py        # Observation functions
│       └── terminations.py        # Termination conditions
│
├── rl_framework/                   # RL Algorithm Implementation
│   └── rsl_rl/                    # RSL-RL library (NVIDIA's RL framework)
│       ├── algorithms/
│       │   ├── ppo.py             # ★ PPO algorithm implementation
│       │   └── distillation.py    # Policy distillation
│       ├── modules/
│       │   ├── actor_critic.py    # ★ Actor-Critic neural networks
│       │   └── actor_critic_recurrent.py  # LSTM variants
│       ├── runners/
│       │   └── on_policy_runner.py  # ★ Main training loop
│       ├── storage/
│       │   └── rollout_storage.py # ★ Experience buffer
│       ├── networks/
│       │   ├── mlp.py             # Multi-layer perceptron
│       │   └── normalization.py   # Observation normalization
│       └── vecenv_wrapper.py      # IsaacLab environment wrapper
│
├── scripts/                        # Training/Testing Scripts
│   ├── train.py                   # ★ NVIDIA's training script
│   ├── play.py                    # ★ Policy visualization
│   └── cli_args.py                # Command-line arguments
│
├── train_robot_arm.py             # Simple training wrapper
├── run_training.sh                # Training launcher
├── play_trained_policy.sh         # Policy visualization launcher
└── README.md                      # This file

★ = Core RL implementation files
```

## Implementation Components

### 1. Environment Configuration (`lift/lift_env_cfg.py`)

The base configuration defines:

- **Scene**: Robot, cube, table, lighting
- **Actions**: Arm control + gripper open/close
- **Observations**: Joint positions/velocities, object position, target position
- **Commands**: Random target pose generation for placement
- **Rewards**:
  - `reaching_object`: Encourages end-effector to reach cube
  - `lifting_object`: Rewards lifting cube above threshold (0.04m)
  - `object_goal_tracking`: Rewards moving cube toward goal
  - `object_goal_tracking_fine_grained`: Precise placement reward
  - `action_rate`: Penalizes large actions (smoothness)
  - `joint_vel`: Penalizes high joint velocities
- **Terminations**: Episode ends if cube drops below table
- **Curriculum**: Gradually increases action/velocity penalties

**Key parameters:**
- Episode length: 5 seconds
- Control frequency: 50Hz (decimation=2, sim dt=0.01)
- Environments: 4096 parallel simulations

### 2. Robot Configuration (`lift/config/franka/`)

Three control modes available:

#### a) **Joint Position Control** (`joint_pos_env_cfg.py`)
- Direct joint position commands
- Simpler but less intuitive for manipulation

#### b) **IK Relative Control** (`ik_rel_env_cfg.py`) ← **CURRENTLY ACTIVE**
- Commands are **relative** pose changes (dx, dy, dz, droll, dpitch, dyaw)
- More natural for learning incremental movements
- Uses Differential IK with Damped Least Squares (DLS)

#### c) **IK Absolute Control** (`ik_abs_env_cfg.py`)
- Commands are **absolute** target poses in world frame
- Good for precise positioning

**Cube setup:**
- Initial position: [0.5, 0, 0.055] (on table, in front of robot)
- Randomization: x±0.1m, y±0.25m
- Size: 0.8x scaled DexCube

**Target placement:**
- X: 0.4 to 0.6m from robot
- Y: -0.25 to 0.25m (left/right)
- Z: 0.055m (on table surface) ← **Modified to place on table, not mid-air**

### 3. MDP Components (`lift/mdp/`)

#### **Rewards** (`rewards.py`)
```python
def object_ee_distance(env, std):
    """Reward reaching the object with end-effector (tanh kernel)"""

def object_is_lifted(env, minimal_height):
    """Binary reward for lifting cube above threshold"""

def object_goal_distance(env, std, minimal_height, command_name):
    """Reward for moving lifted cube toward goal pose"""
```

#### **Observations** (`observations.py`)
```python
def object_position_in_robot_root_frame(env):
    """Cube position relative to robot base"""

def joint_pos_rel(env):
    """Joint positions relative to default"""

def joint_vel_rel(env):
    """Joint velocities"""
```

#### **Terminations** (`terminations.py`)
- Time out (5 seconds)
- Object dropping below table

### 4. PPO Algorithm (`lift/config/franka/agents/rsl_rl_ppo_cfg.py`)

**Training hyperparameters:**
- `num_steps_per_env = 24`: Collect 24 timesteps per environment
- `max_iterations = 1500`: Total training iterations
- `save_interval = 200`: Save checkpoint every 200 iterations
- `num_learning_epochs = 5`: 5 epochs per policy update
- `num_mini_batches = 4`: Split data into 4 mini-batches

**Network architecture:**
- Actor (policy): [256, 128, 64] with ELU activation
- Critic (value): [256, 128, 64] with ELU activation

**PPO settings:**
- Learning rate: 1e-4 with adaptive schedule
- Clip param: 0.2
- Entropy coefficient: 0.006
- GAE: γ=0.98, λ=0.95

**Per iteration:**
- Timesteps collected: 24 × 4096 = 98,304
- Gradient updates: 5 epochs × 4 batches = 20 updates
- Every 200 iterations = ~19.7M timesteps between saves

## How the Task Works

### Task Flow:
1. **Reset**: Cube spawns at random position on table
2. **Command**: Generate random target pose on table
3. **Policy**: Robot observes state and decides arm/gripper actions
4. **Execution**:
   - Move end-effector toward cube
   - Close gripper to grasp
   - Lift cube off table
   - Move to target location
   - Open gripper to release
5. **Rewards**: Agent gets rewards for reaching, lifting, and placing
6. **Termination**: Episode ends after 5s or if cube falls

### Learning Process:
- Agent learns **when to close/open gripper** (binary action)
- Agent learns **smooth arm movements** (penalized for jerky actions)
- Agent learns **complete pick-and-place sequence** (no explicit stages)
- Curriculum increases penalties to enforce smooth policies

## Running the Code

### Train:
```bash
./run_training.sh
# Or with resume:
./run_training.sh --resume
```

### Visualize trained policy:
```bash
./play_trained_policy.sh
```

### Monitor training:
```bash
tensorboard --logdir logs/rsl_rl/Isaac-Lift-Cube-Franka-IK-Rel-v0
```

## Key Implementation Insights

1. **No explicit pick/place actions**: The policy learns the full sequence through reward shaping
2. **Parallel simulation**: 4096 environments = massive data collection speed
3. **Relative IK**: More sample-efficient than joint control for manipulation
4. **Reward shaping**: Critical for learning - three different rewards guide behavior:
   - Reach (early learning)
   - Lift (intermediate)
   - Place (final objective)
5. **Curriculum**: Starts with weak action penalties, gradually increases for smoother policies
6. **Binary gripper**: Simple open/close, not continuous force control

## Dependencies

This code requires:
- Isaac Sim 4.0+
- IsaacLab
- RSL RL (PPO implementation)
- PyTorch

## What Makes This Different from Your Current Setup?

Your current `lift_task/` directory has the **same NVIDIA code BUT with modifications**:
- Modified placement height to place cubes ON table (not mid-air)
- Changed save frequency to reduce checkpoint spam

This `fullcode_nvidia_lift/` is:
- ✅ **Pure NVIDIA original** (reference copy for study)
- ✅ **Complete RL implementation** included (PPO algorithm, training loop, etc.)
- ✅ **All source code** from RSL-RL library (not just wrapper)

## Files Copied From

- **Task code**: `/home/kenpeter/work/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/`
- **RL framework**: `/home/kenpeter/anaconda3/envs/isaaclab_env/lib/python3.11/site-packages/rsl_rl/`
- **Training scripts**: `/home/kenpeter/work/IsaacLab/scripts/reinforcement_learning/rsl_rl/`

This is the **official NVIDIA implementation** from IsaacLab + RSL-RL.
