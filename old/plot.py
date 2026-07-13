import numpy as np
import matplotlib.pyplot as plt
import json
import os
from logger import Logger
from pathlib import Path
from datetime import datetime
import pytz
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
"""
 To compare EWA-ODT vs. ODT, you should focus on key metrics related to learning efficiency, performance stability, and return improvement.

📊 Figure 1: Normalized Return vs. Training Steps
    X-axis: Training Steps (Number of Online Updates)
    Y-axis: Normalized Return
    Purpose: This will show how quickly EWA-ODT adapts compared to ODT. If EWA-ODT outperforms ODT, it should reach a higher return faster.
📊 Figure 2: Return vs. Number of Online Rollouts
    X-axis: Number of Online Rollouts
    Y-axis: Return (Mean Episode Return)
    Purpose: This figure shows how well both models adapt as they see more data. If EWA-ODT is better at using past experiences, 
    it should show higher returns with fewer rollouts.
📊 Figure 3: Variance in Action Selection Over Time
    X-axis: Training Iterations
    Y-axis: Action Entropy (Diversity in Action Selection)
    Purpose: Since EWA modifies attention weights based on past action attraction, we expect it to create more structured and 
    stable decision-making over time. This plot will show if EWA reduces instability in action choices.
"""


# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')
now = datetime.now(tz=us_eastern).strftime("%Y.%m.%d/%H")

# 📌 Set plot style for top-tier ML conferences (NeurIPS, ICLR, ICML)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "legend.fontsize": 10
})

# 🔹 Path to experiment results directory
RESULTS_DIR = "./exp_results"

# 🔹 Function to extract data from TensorBoard event files
def extract_tb_data(log_dir, metric_name):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    if metric_name not in event_acc.Tags()["scalars"]:
        print(f"⚠️ Metric {metric_name} not found in {log_dir}")
        return None, None

    events = event_acc.Scalars(metric_name)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    
    return steps, values

# 🔹 Function to plot results with mean and standard deviation
def plot_results(data, title, x_label, y_label, filename):
    plt.figure(figsize=(7, 5))
    
    for label, runs in data.items():
        x_all, y_all = zip(*runs)
        x_all = np.array(x_all)
        y_all = np.array(y_all)
        
        mean_y = np.mean(y_all, axis=0)
        std_y = np.std(y_all, axis=0)

        plt.plot(x_all[0], mean_y, label=label, linewidth=2)
        plt.fill_between(x_all[0], mean_y - std_y, mean_y + std_y, alpha=0.2)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved plot: {filename}")
    plt.show()

# 🔹 Define figures to plot
figures = {
    "Normalized Return vs. Training Steps": {
        "metric": "evaluation/return_mean_gm",
        "x_label": "Training Steps",
        "y_label": "Normalized Return",
        "filename": "fig_normalized_return_vs_steps.png",
    },
    "Return vs. Number of Online Rollouts": {
        "metric": "aug_traj/return",
        "x_label": "Number of Online Rollouts",
        "y_label": "Return",
        "filename": "fig_return_vs_online_rollouts.png",
    },
    "Action Entropy vs. Training Iterations": {
        "metric": "training/entropy",
        "x_label": "Training Iterations",
        "y_label": "Action Entropy",
        "filename": "fig_action_entropy_vs_training.png",
    },
}

# 🔹 Extract and plot data
for fig_name, config in figures.items():
    data = {}
    
    # Scan experiment directories
    for exp_name in os.listdir(RESULTS_DIR):
        exp_path = os.path.join(RESULTS_DIR, exp_name)
        if os.path.isdir(exp_path):
            tb_files = [f for f in os.listdir(exp_path) if "tfevents" in f]
            if tb_files:
                tb_path = os.path.join(exp_path, tb_files[0])
                x, y = extract_tb_data(tb_path, config["metric"])
                if x is not None and y is not None:
                    if exp_name not in data:
                        data[exp_name] = []
                    data[exp_name].append((x, y))
    
    # Plot the figure
    if data:
        plot_results(data, fig_name, config["x_label"], config["y_label"], config["filename"])

# ##########################################

log_dir = "./exp/2025.02.23/033428-ewa-hopper-medium-v2"
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

expert_returns = {"hopper": 3600, "walker2d": 5000, "halfcheetah": 6000, "antmaze": 1}

# Extract evaluation return mean values
return_means = [scalar.value for scalar in event_acc.Scalars("evaluation/return_mean_gm")]
return_std = [scalar.value for scalar in event_acc.Scalars("evaluation/return_std_gm")]

# Compute final metrics
avg_return = np.mean(return_means)
std_return = np.mean(return_std)

print(f"Average Return: {avg_return}, Std Dev: {std_return}")

expert_returns = {"hopper": 3600, "walker2d": 5000, "halfcheetah": 6000, "antmaze": 1}
env = "hopper"  # Change this based on your experiment
normalized_return = avg_return / expert_returns[env]
print(f"Normalized Return for {env}: {normalized_return}")


