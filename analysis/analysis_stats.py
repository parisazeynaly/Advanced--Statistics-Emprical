import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

df = pd.read_csv("results/multienv.csv")
os.makedirs("figures", exist_ok=True)


# ---- Cohen's d function ----
def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_d(d):
    """Label the effect size."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    elif ad < 1.2:
        return "large"
    else:
        return "very large"


# ---- Effect sizes: Thompson vs each other strategy, per environment ----
print("="*70)
print("EFFECT SIZES (Cohen's d) — based on total_reward (learning speed)")
print("="*70)

for env in df["environment"].unique():
    env_data = df[df["environment"] == env]
    thompson = env_data[env_data["strategy"] == "thompson"]["total_reward"].values

    print(f"\n{env}:")
    for other in ["epsilon_greedy", "softmax"]:
        other_data = env_data[env_data["strategy"] == other]["total_reward"].values
        d = cohens_d(thompson, other_data)
        t_stat, p_value = stats.ttest_ind(thompson, other_data)
        print(f"  thompson vs {other:15s}: "
              f"d = {d:6.2f} ({interpret_d(d):11s}), "
              f"p = {p_value:.2e}")


# ---- Box plots: one panel per environment ----
environments = df["environment"].unique()
n_envs = len(environments)

fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 5))

for ax, env in zip(axes, environments):
    env_data = df[df["environment"] == env]

    # gather total_reward for each strategy
    data_to_plot = [env_data[env_data["strategy"] == s]["total_reward"].values
                    for s in ["epsilon_greedy", "softmax", "thompson"]]

    ax.boxplot(data_to_plot, tick_labels=["ε-greedy", "softmax", "thompson"])
    ax.set_title(env)
    ax.set_ylabel("Total reward (learning speed)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

plt.suptitle("Distribution of Learning Speed Across 30 Seeds", fontsize=14)
plt.tight_layout()
plt.savefig("figures/boxplots.png", dpi=150, bbox_inches="tight")
print("\n\nSaved figures/boxplots.png")
plt.close()

# ---- Zoomed box plots: exclude epsilon_greedy to see softmax vs thompson ----
fig, axes = plt.subplots(1, len(environments), figsize=(5 * len(environments), 5))

for ax, env in zip(axes, environments):
    env_data = df[df["environment"] == env]

    # only softmax and thompson (drop the extreme epsilon_greedy)
    data_to_plot = [env_data[env_data["strategy"] == s]["total_reward"].values
                    for s in ["softmax", "thompson"]]

    ax.boxplot(data_to_plot, tick_labels=["softmax", "thompson"])
    ax.set_title(env)
    ax.set_ylabel("Total reward (learning speed)")
    ax.grid(True, alpha=0.3)

plt.suptitle("Learning Speed: Softmax vs Thompson (zoomed, ε-greedy excluded)",
             fontsize=14)
plt.tight_layout()
plt.savefig("figures/boxplots_zoomed.png", dpi=150, bbox_inches="tight")
print("Saved figures/boxplots_zoomed.png")
plt.close()