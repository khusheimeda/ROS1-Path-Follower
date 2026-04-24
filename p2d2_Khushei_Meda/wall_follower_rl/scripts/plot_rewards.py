#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_rewards.py  --  Plot accumulated reward vs episodes for Q-learning and SARSA.

Usage:
    python scripts/plot_rewards.py

Reads from q_tables/ directory, saves figure to q_tables/reward_plot.png
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_rewards(path):
    """Load episode rewards from CSV, return (episodes[], rewards[])."""
    episodes = []
    rewards = []
    if not os.path.isfile(path):
        print('File not found: {}'.format(path))
        return episodes, rewards
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            episodes.append(int(row[0]))
            rewards.append(float(row[1]))
    return episodes, rewards


def smooth(values, window=20):
    """Simple moving average for cleaner plot."""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        smoothed.append(np.mean(values[lo:i+1]))
    return smoothed


def main():
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'q_tables')

    ql_path = os.path.join(base_dir, 'q_table_qlearning_rewards.csv')
    sarsa_path = os.path.join(base_dir, 'q_table_sarsa_rewards.csv')
    out_path = os.path.join(base_dir, 'reward_plot.png')

    ql_eps, ql_rewards = load_rewards(ql_path)
    sarsa_eps, sarsa_rewards = load_rewards(sarsa_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Q-learning ---
    if ql_rewards:
        ax.plot(ql_eps, ql_rewards, alpha=0.2, color='blue')
        ax.plot(ql_eps, smooth(ql_rewards, 20), color='blue', linewidth=2,
                label='Q-Learning (smoothed)')

        # If hyperparameters changed mid-training, mark it
        # Change 300 to whatever episode YOUR change happened at
        ql_change_ep = 300
        if len(ql_eps) > ql_change_ep:
            ax.axvline(x=ql_change_ep, color='blue', linestyle='--', alpha=0.5,
                       label='Q-Learning: ε changed (ep {})'.format(ql_change_ep))

    # --- SARSA ---
    if sarsa_rewards:
        ax.plot(sarsa_eps, sarsa_rewards, alpha=0.2, color='red')
        ax.plot(sarsa_eps, smooth(sarsa_rewards, 20), color='red', linewidth=2,
                label='SARSA (smoothed)')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Accumulated Reward', fontsize=12)
    ax.set_title('Q-Learning vs SARSA: Accumulated Reward per Episode', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved plot to {}'.format(out_path))
    plt.show()


if __name__ == '__main__':
    main()
