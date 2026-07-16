import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("results/all_results.csv")

# Basic overview
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst few rows:")
print(df.head())

print("\nStrategies:", df["strategy"].unique())
print("Map sizes:", df["map_size"].unique())
print("Seeds per group:", df["seed"].nunique())

# Quick sanity check: average final success rate per strategy & map
print("\nAverage final success rate (%):")
summary = df.groupby(["map_size", "strategy"])["final_success"].mean() * 100
print(summary.round(1))

import matplotlib.pyplot as plt

def plot_learning_curves(df, map_name):
    """Plot average reward with 95% confidence bands for each strategy."""
    plt.figure(figsize=(10, 6))

    subset = df[df["map_size"] == map_name]
    window = 100

    for strategy in subset["strategy"].unique():
        strat_data = subset[subset["strategy"] == strategy]

        # pivot: rows = episodes, columns = seeds, values = reward
        pivot = strat_data.pivot(index="episode", columns="seed", values="reward")

        # smooth each seed's curve first
        smoothed = pivot.rolling(window=window, min_periods=1).mean()

        # mean across seeds, and standard error
        mean_curve = smoothed.mean(axis=1)
        std_curve = smoothed.std(axis=1)
        n_seeds = smoothed.shape[1]
        stderr = std_curve / np.sqrt(n_seeds)

        # 95% confidence interval ≈ mean ± 1.96 * standard error
        ci = 1.96 * stderr

        episodes = mean_curve.index

        # plot the mean line
        line = plt.plot(episodes, mean_curve.values, label=strategy)
        color = line[0].get_color()

        # plot the shaded confidence band
        plt.fill_between(episodes,
                         mean_curve.values - ci.values,
                         mean_curve.values + ci.values,
                         color=color, alpha=0.2)

    plt.title(f"Learning Curves with 95% CI on FrozenLake {map_name}")
    plt.xlabel("Episode")
    plt.ylabel(f"Average reward (rolling mean, window={window})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"figures/learning_curve_ci_{map_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close()


# Make the figures folder and create both plots
import os
os.makedirs("figures", exist_ok=True)

plot_learning_curves(df, "4x4")
plot_learning_curves(df, "8x8")

from scipy import stats

def compare_strategies(df, map_name):
    """Statistically compare strategies by total training reward per seed."""
    print(f"\n{'='*50}")
    print(f"Statistical comparison on FrozenLake {map_name}")
    print(f"{'='*50}")

    subset = df[df["map_size"] == map_name]

    # total reward per seed for each strategy (area under learning curve)
    totals = {}
    for strategy in subset["strategy"].unique():
        strat_data = subset[subset["strategy"] == strategy]
        # sum of rewards across all episodes, per seed
        per_seed_total = strat_data.groupby("seed")["reward"].sum()
        totals[strategy] = per_seed_total
        print(f"\n{strategy}:")
        print(f"  mean total reward = {per_seed_total.mean():.1f}")
        print(f"  std  = {per_seed_total.std():.1f}")

    # pairwise t-tests between strategies
    strategies = list(totals.keys())
    print(f"\n{'-'*50}")
    print("Pairwise t-tests (is the difference significant?):")
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            t_stat, p_value = stats.ttest_ind(totals[s1], totals[s2])
            significant = "YES ✓" if p_value < 0.05 else "no"
            print(f"  {s1} vs {s2}: p = {p_value:.4g}  → significant? {significant}")


compare_strategies(df, "4x4")
compare_strategies(df, "8x8")