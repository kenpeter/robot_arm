# Guide to the RL Implementation Code

This guide highlights the **key RL algorithm files** you should study to understand how PPO training works.

## Start Here: The Core Algorithm Files

### 1. **Training Loop** (`rl_framework/rsl_rl/runners/on_policy_runner.py`)

This is the **main training orchestrator**. Read this first!

**Key methods:**
- `__init__()`: Sets up environment, algorithm, storage
- `learn()`: Main training loop (collect → compute advantages → update policy)
- `save()` / `load()`: Checkpoint management

**Flow:**
```python
def learn(num_iterations):
    for iteration in range(num_iterations):
        # 1. Collect rollouts
        for step in range(num_steps_per_env):
            actions = policy(obs)
            obs, rewards, dones = env.step(actions)
            storage.add_transitions(...)

        # 2. Compute returns/advantages
        storage.compute_returns()

        # 3. Update policy
        alg.update()

        # 4. Log & save
        log_metrics()
        if iteration % save_interval == 0:
            save_checkpoint()
```

---

### 2. **PPO Algorithm** (`rl_framework/rsl_rl/algorithms/ppo.py`)

This implements the **actual PPO update**.

**Key methods:**
- `act()`: Sample action from current policy
- `update()`: Perform PPO policy update
- `evaluate()`: Evaluate actions for gradient computation

**Core PPO logic:**
```python
def update():
    for epoch in range(num_learning_epochs):
        for minibatch in data_generator:
            # Re-evaluate actions
            values, log_probs, entropy = actor_critic.evaluate(obs, actions)

            # Compute ratio
            ratio = exp(log_probs - old_log_probs)

            # PPO clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = clip(ratio, 1-eps, 1+eps) * advantages
            policy_loss = -min(surr1, surr2).mean()

            # Value loss
            value_loss = (values - returns).pow(2).mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
```

**This is where the magic happens!**

---

### 3. **Actor-Critic Network** (`rl_framework/rsl_rl/modules/actor_critic.py`)

The **neural network** that represents the policy and value function.

**Key components:**
- `ActorCritic`: Main class
  - `actor`: Policy network (obs → action mean)
  - `critic`: Value network (obs → value estimate)
  - `std`: Action noise (learned parameter)

**Key methods:**
```python
def act(observations):
    """Sample action from policy (used during rollout collection)"""
    action_mean = self.actor(observations)
    action = action_mean + self.std * torch.randn_like(action_mean)
    value = self.critic(observations)
    log_prob = gaussian_log_prob(action, action_mean, self.std)
    return action, value, log_prob

def evaluate(observations, actions):
    """Evaluate actions (used during PPO update)"""
    action_mean = self.actor(observations)
    value = self.critic(observations)
    log_prob = gaussian_log_prob(actions, action_mean, self.std)
    entropy = gaussian_entropy(self.std)
    return value, log_prob, entropy
```

**Network architecture** (from config):
- Input: Observation vector
- Actor: [256, 128, 64] → action_dim
- Critic: [256, 128, 64] → 1
- Activation: ELU

---

### 4. **Rollout Storage** (`rl_framework/rsl_rl/storage/rollout_storage.py`)

Stores collected experience and **computes advantages using GAE**.

**Key methods:**
- `add_transitions()`: Store obs, action, reward, value, log_prob
- `compute_returns()`: Compute GAE advantages and returns
- `mini_batch_generator()`: Yield minibatches for training

**GAE computation:**
```python
def compute_returns(last_values, gamma, lam):
    """Generalized Advantage Estimation"""
    advantage = 0
    for step in reversed(range(num_steps)):
        # TD error
        delta = rewards[step] + gamma * values[step+1] * (1-dones[step]) - values[step]

        # GAE recursion
        advantage = delta + gamma * lam * (1-dones[step]) * advantage

        # Store
        self.advantages[step] = advantage
        self.returns[step] = advantage + values[step]
```

**Why GAE?**
- Balances bias vs. variance
- γ (gamma): discount factor for future rewards
- λ (lambda): controls how much to trust value function

---

### 5. **Environment Wrapper** (`rl_framework/rsl_rl/vecenv_wrapper.py`)

Wraps IsaacLab environment to work with RSL-RL.

**Key functionality:**
- Converts between IsaacLab and RSL-RL interfaces
- Handles observation/action formatting
- Manages episode resets across parallel environments

---

## Supporting Files

### Networks
- `rl_framework/rsl_rl/networks/mlp.py`: Basic MLP implementation
- `rl_framework/rsl_rl/networks/normalization.py`: Running mean/std for obs normalization

### Training Scripts
- `scripts/train.py`: Command-line training script (uses OnPolicyRunner)
- `scripts/play.py`: Load checkpoint and visualize policy
- `scripts/cli_args.py`: Argument parsing

---

## Reading Order (Recommended)

If you want to understand the full RL implementation:

1. **Start**: `rl_framework/rsl_rl/modules/actor_critic.py`
   - Understand the policy/value networks

2. **Then**: `rl_framework/rsl_rl/storage/rollout_storage.py`
   - See how experience is stored and advantages computed

3. **Next**: `rl_framework/rsl_rl/algorithms/ppo.py`
   - Learn the PPO update equation

4. **Finally**: `rl_framework/rsl_rl/runners/on_policy_runner.py`
   - See how everything fits together in the training loop

5. **Bonus**: `scripts/train.py`
   - Understand how to launch training

---

## Key Concepts to Understand

### PPO (Proximal Policy Optimization)
- **On-policy**: Learns from recent experience
- **Clipping**: Prevents large policy updates (stable learning)
- **Multiple epochs**: Reuses data for sample efficiency
- **Trust region**: Keeps new policy close to old policy

### Actor-Critic
- **Actor**: Policy (chooses actions)
- **Critic**: Value function (predicts future rewards)
- **Separation**: Actor explores, critic provides baseline

### GAE (Generalized Advantage Estimation)
- **Advantage**: How much better is action vs. average?
- **Returns**: Total future reward from this state
- **GAE**: Exponentially-weighted average of n-step advantages

### Parallel Environments
- **Vectorization**: Run 4096 robots in parallel
- **Speed**: Collect 98k transitions in seconds
- **GPU**: Fully parallelized on GPU

---

## Code Flow Diagram

```
train.py
   │
   └─→ OnPolicyRunner.learn()
         │
         ├─→ [Data Collection]
         │     │
         │     ├─→ ActorCritic.act(obs)
         │     │     └─→ returns: actions, values, log_probs
         │     │
         │     ├─→ env.step(actions)
         │     │     └─→ returns: obs, rewards, dones
         │     │
         │     └─→ RolloutStorage.add_transitions()
         │
         ├─→ [Advantage Computation]
         │     │
         │     └─→ RolloutStorage.compute_returns()
         │           └─→ GAE algorithm
         │
         ├─→ [Policy Update]
         │     │
         │     └─→ PPO.update()
         │           │
         │           ├─→ ActorCritic.evaluate(obs, actions)
         │           ├─→ Compute PPO loss
         │           └─→ Backprop & optimizer step
         │
         └─→ [Logging & Checkpointing]
               ├─→ TensorBoard logging
               └─→ Save checkpoint
```

---

## Hyperparameters Explained

From `lift/config/franka/agents/rsl_rl_ppo_cfg.py`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_steps_per_env` | 24 | Timesteps to collect per env before update |
| `num_learning_epochs` | 5 | How many times to reuse collected data |
| `num_mini_batches` | 4 | Split data into N batches per epoch |
| `learning_rate` | 1e-4 | Adam optimizer step size |
| `gamma` | 0.98 | Discount factor (future reward importance) |
| `lam` | 0.95 | GAE lambda (bias/variance tradeoff) |
| `clip_param` | 0.2 | PPO clipping epsilon (trust region) |
| `entropy_coef` | 0.006 | Exploration bonus weight |
| `value_loss_coef` | 1.0 | Value function loss weight |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |

**Total gradient updates per iteration**: 5 epochs × 4 batches = **20 updates**

---

## Debugging Tips

### If training is unstable:
- Check `entropy`: Should decrease slowly (if too fast → reduce lr)
- Check `kl_divergence`: Should stay near `desired_kl` (0.01)
- Check `value_loss`: Should decrease steadily

### If learning is too slow:
- Increase `num_envs` (more data)
- Decrease `gamma` (focus on short-term rewards)
- Increase `learning_rate`
- Check reward scale (rewards too small?)

### If policy is jerky:
- Increase `action_rate` penalty weight
- Increase `joint_vel` penalty weight
- Decrease `learning_rate` (slower updates)

---

## Further Reading

**Papers:**
- [PPO](https://arxiv.org/abs/1707.06347): Proximal Policy Optimization
- [GAE](https://arxiv.org/abs/1506.02438): Generalized Advantage Estimation
- [Actor-Critic](https://arxiv.org/abs/1602.01783): Asynchronous Methods for Deep RL

**Code:**
- [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl): Original repository
- [IsaacLab](https://docs.isaac-sim.com/): NVIDIA's robot learning framework

---

Happy studying! 🚀
