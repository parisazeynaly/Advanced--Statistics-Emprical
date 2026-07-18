import numpy as np
import pandas as pd
from train import train, evaluate

# Best parameters from the sensitivity analysis
strategy_params = {
    "epsilon_greedy": {"epsilon": 0.3, "temperature": 1.0},
    "softmax":        {"epsilon": 0.3, "temperature": 0.1},
    "thompson":       {"epsilon": 0.3, "temperature": 1.0},
}

# Environments to test (FrozenLake in both sizes counts as two)
environments = [
    {"env_name": "FrozenLake", "map_name": "4x4", "label": "FrozenLake-4x4"},
    {"env_name": "FrozenLake", "map_name": "8x8", "label": "FrozenLake-8x8"},
    {"env_name": "CliffWalking", "map_name": "4x4", "label": "CliffWalking"},
    {"env_name": "Taxi", "map_name": "4x4", "label": "Taxi"},
]

n_seeds = 30
n_episodes = 5000

all_rows = []

for env_config in environments:
    env_name = env_config["env_name"]
    map_name = env_config["map_name"]
    label = env_config["label"]

    for strategy, params in strategy_params.items():
        print(f"Running {strategy} on {label}...")
        for seed in range(n_seeds):
            q_table, rewards = train(
                strategy=strategy, env_name=env_name, map_name=map_name,
                seed=seed, epsilon=params["epsilon"],
                temperature=params["temperature"], n_episodes=n_episodes,
            )
            total_reward = np.sum(rewards)
            final = evaluate(q_table, env_name=env_name, map_name=map_name)

            all_rows.append({
                "environment": label,
                "strategy": strategy,
                "seed": seed,
                "total_reward": total_reward,
                "final_reward": final,
            })

df = pd.DataFrame(all_rows)
df.to_csv("results/multienv.csv", index=False)
print("\nDone! Saved to results/multienv.csv")
print("Total rows:", len(df))