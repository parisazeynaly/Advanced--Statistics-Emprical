import numpy as np
import pandas as pd
from train import train, evaluate

# ---- Experiment settings ----
strategies = ["epsilon_greedy", "softmax", "thompson"]
map_sizes = ["4x4", "8x8"]
n_seeds = 30
n_episodes = 5000

# Sensible knob values for each strategy (chosen for fair comparison)
strategy_params = {
    "epsilon_greedy": {"epsilon": 0.3, "temperature": 1.0},
    "softmax":        {"epsilon": 0.3, "temperature": 0.1},
    "thompson":       {"epsilon": 0.3, "temperature": 1.0},
}

# ---- Storage for all results ----
all_rows = []

# ---- Run every combination ----
for map_name in map_sizes:
    for strategy in strategies:
        params = strategy_params[strategy]
        print(f"Running {strategy} on {map_name}...")

        for seed in range(n_seeds):
            q_table, rewards = train(
                strategy=strategy,
                n_episodes=n_episodes,
                seed=seed,
                epsilon=params["epsilon"],
                temperature=params["temperature"],
                map_name=map_name,
            )

            final_success = evaluate(q_table, map_name=map_name)

            # store one row per episode
            for episode, reward in enumerate(rewards):
                all_rows.append({
                    "map_size": map_name,
                    "strategy": strategy,
                    "seed": seed,
                    "episode": episode,
                    "reward": reward,
                    "final_success": final_success,
                })

# ---- Save to CSV ----
df = pd.DataFrame(all_rows)
df.to_csv("results/all_results.csv", index=False)
print("\nDone! Saved results to results/all_results.csv")
print("Total rows:", len(df))