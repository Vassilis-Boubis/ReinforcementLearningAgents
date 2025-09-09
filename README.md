# Reinforcement Learning Agents

This repository contains code for multiple RL concepts, such as Dynamic Programming, Tabular Reinforcement Learning, Deep Reinforcement Learning through Deep Q Networks (DQN), and Deep Deterministic Policy Gradient (DDPG).
The goal was to implement, analyze, and extend reinforcement learning (RL) algorithms in both discrete and continuous environments using PyTorch.

## Structure


### **Dynamic Programming** 
- Implemented **Value Iteration** and **Policy Iteration** algorithms.
- Verified correctness using toy MDPs (example with "Frog on a Rock").
- Key file: `mdp_solver.py`

### **Tabular Reinforcement Learning**
- Implemented and evaluated agent performance on `FrozenLake8x8-v1` (deterministic and slippery variants).
  - ε-greedy action selection.
  - **Q-Learning**
  - **On-policy Every-Visit Monte Carlo**
- Explored and compared hyperparameter profiles with different choices for `gamma`.
- Key files: `agents.py`, `train_q_learning.py`, `train_monte_carlo.py`

### **Deep Reinforcement Learning**
- Implemented **Deep Q-Networks (DQN)** with:
  - Epsilon scheduling strategies: including exponential and linear decay.
  - Target network updates
  - Gradient-based updates
- Compared DQN with a tabular discrete agent on the `MountainCar-v0` environment.
- Analyzed the DQN loss and learning behavior.
- Key files: `agents.py`, `train_dqn.py`

### **Continuous Control with DDPG** 
- Implemented the **Deep Deterministic Policy Gradient (DDPG)** algorithm for continuous action spaces.
- Trained agents on the `Racetrack` environment from `highway-env`.
- Tuned actor/critic network architectures to achieve competitive performance. Achieved more stable and increased results from the given baseline.
- Key files: `agents.py`, `train_ddpg.py`, `evaluate_ddpg.py`

### **Algorithm Extensions on Tabular Reinforcement Learning**
- Implemented Upper Confidence Bound (UCB) as an exploration strategy in both tabular (DiscereRL) and function approximation (DQN) settings, and compared it against ε-greedy.
- Showed that while ε-greedy explores randomly, UCB leverages a confidence bonus to encourage exploration of less-visited actions, resulting in more intelligent exploration paths.
- Results indicated that UCB provides more stable long-term performance in the tabular setting, though both strategies suffer from high variance and low returns. In the DQN setting, UCB initially learns slower but eventually outperforms ε-greedy by achieving lower variance and better average returns (-107 vs -110).
- Key files: `train_ddpg.py`, `train_discreterl.py`

---

## Setup Instructions

**Create and activate a conda environment and install dependencies:**

```bash
conda create -n rl_course python=3.7
conda activate rl_course
```

```bash
pip install -e .
```
