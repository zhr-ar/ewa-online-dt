import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import pytz

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')

# Define the base paths for EWA and ODT results (update paths as needed)
ewa_base_path = "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/"
odt_base_path = "/home/ubuntu/online_decision_transformer/online-dt-main/"

ewa_paths = [
    f"{ewa_base_path}exp_results/normalized_return_vs_steps_walker2d-medium-replay-v2_20250225-134313/134320_ewa_walker2d-medium-replay-v2",
    f"{ewa_base_path}exp_results/normalized_return_vs_steps_hopper-medium-v2_20250225-171230/171236_ewa_hopper-medium-v2",
    f"{ewa_base_path}exp_results/normalized_return_vs_steps_halfcheetah-medium-expert-v2_20250225-083235/083249_ewa_halfcheetah-medium-expert-v2"
]

odt_paths = [
    f"{odt_base_path}exp_results/normalized_return_vs_steps_walker2d-medium-replay-v2_20250225-114320/114329-odt-walker2d-medium-replay-v2",
    f"{odt_base_path}exp_results/normalized_return_vs_steps_hopper-medium-v2_20250225-134351/134400-odt-hopper-medium-v2",
    f"{odt_base_path}exp_results/normalized_return_vs_steps_halfcheetah-medium-expert-v2_20250225-091253/091305-odt-halfcheetah-medium-expert-v2"
]

# Define metrics to compare
metrics = [
    "evaluation/length_mean_gm",
    "evaluation/length_std_gm",
    "evaluation/return_mean_gm",
    "evaluation/return_std_gm",
    "evaluation/return_vs_samples",
    "aug_traj/length",
    "aug_traj/return"
]

# Function to load event logs
def load_event_logs(folder_path, metric_name):
    event_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.startswith("events.out.tfevents")]

    if not event_files:
        print(f"Warning: No event file found in {folder_path}")
        return None

    # Sort event files by timestamp to ensure proper ordering
    event_files.sort()

    steps = []
    values = []
    for event_file in event_files:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                if v.tag == metric_name:
                    steps.append(e.step)
                    values.append(v.simple_value)

    if not steps or not values:
        print(f"Warning: No data found for {metric_name} in any event file in {folder_path}")
        return None

    # Convert to numpy arrays and sort by steps
    steps = np.array(steps)
    values = np.array(values)
    sort_idx = np.argsort(steps)
    steps = steps[sort_idx]
    values = values[sort_idx]

    return steps, values

# Function to plot comparisons
def plot_comparison(env, ewa_folder, odt_folder, metric_name):
    ewa_steps, ewa_values = load_event_logs(ewa_folder, metric_name)
    odt_steps, odt_values = load_event_logs(odt_folder, metric_name)

    # Check if any of the steps or values are None
    if ewa_steps is None or odt_steps is None:
        print(f"Skipping plot for {env} - {metric_name} due to missing data.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(ewa_steps, ewa_values, label="EWA-ODT", linestyle="-", marker="o")
    plt.plot(odt_steps, odt_values, label="ODT", linestyle="--", marker="s")
    plt.xlabel("Training Steps")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"{metric_name.replace('_', ' ').title()} - {env}")
    plt.legend()
    plt.grid()

    # Create a directory structure with date and time
    now = datetime.now(tz=us_eastern)
    exp_folder = f"figures/{now.strftime('%Y.%m.%d')}/{env}"
    os.makedirs(exp_folder, exist_ok=True)

    # Save the figure
    plt.savefig(f"{exp_folder}/{now.strftime('%H%M%S')}_{metric_name.replace('/', '_')}.png")
    plt.close()  # Close the figure to free up memory

# Loop through environments and generate plots
for ewa_folder, odt_folder in zip(ewa_paths, odt_paths):
    env_name = ewa_folder.split("/")[-1].split("_")[-1]  # Extract env name
    for metric in metrics:
        plot_comparison(env_name, ewa_folder, odt_folder, metric)
