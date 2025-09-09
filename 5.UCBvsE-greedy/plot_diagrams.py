import pandas as pd
import matplotlib.pyplot as plt

c_values = ["1"]
# c_values = ["01", "05", "1", "2", "5", "10"]

filenames = ["eval_returns_dqn_epsilon_greedy.csv"]
filenames += [f"eval_returns_dqn_c{c}_ucb.csv" for c in c_values]

plt.figure(figsize=(10, 6))

labels = ["epsilon-greedy"] + [f"c = {c}" for c in c_values]

for label, file in zip(labels, filenames):
    try:
        returns = pd.read_csv(file, header=None).squeeze()
        plt.plot(returns, label=label)
    except Exception as e:
        print(f"Failed to read {file}: {e}")

plt.title("DQN Performance: Epsilon-Greedy vs UCB Exploration")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend(title="Exploration Strategy")
plt.tight_layout()
plt.show()


c_values = ["1"]
# c_values = ["01", "05", "1", "2", "5", "10"]

filenames = ["eval_returns_discrete_epsilon_greedy.csv"]
filenames += [f"eval_returns_discrete_c{c}_ucb.csv" for c in c_values]

plt.figure(figsize=(10, 6))

labels = ["epsilon-greedy"] + [f"c = {c}" for c in c_values]

for label, file in zip(labels, filenames):
    try:
        returns = pd.read_csv(file, header=None).squeeze()
        plt.plot(returns, label=label)
    except Exception as e:
        print(f"Failed to read {file}: {e}")

plt.title("Discrete Performance: Epsilon-Greedy vs UCB Exploration")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend(title="Exploration Strategy")
plt.tight_layout()
plt.show()