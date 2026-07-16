import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/multienv.csv")
os.makedirs("figures", exist_ok=True)

# ---- Summary table: final reward per strategy per environment ----
print("="*60)
print("FINAL POLICY QUALITY (avg final reward across 30 seeds)")
print("="*60)
summary = df.groupby(["environment", "strategy"])["final_reward"].agg(["mean", "std"]).round(2)
print(summary)

print("\n" + "="*60)
print("LEARNING SPEED (avg total reward / area under curve)")
print("="*60)
summary2 = df.groupby(["environment", "strategy"])["total_reward"].agg(["mean", "std"]).round(1)
print(summary2)