import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import pytz

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')

# Define the base path for results
base_paths = [
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.22",
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.23"
]


# Get all folders for each environment from both base paths
envs = {
    "halfcheetah-medium-expert-v2": [],
    "walker2d-medium-replay-v2": [],
    "hopper-medium-v2": []
}

for base_path in base_paths:
    for env_name in envs.keys():
        env_folders = [f for f in os.listdir(base_path) if env_name in f]
        # Add the base path to each folder name
        env_folders = [(base_path, f) for f in env_folders]
        envs[env_name].extend(env_folders)

# Define metrics to plot
metrics = [
    "evaluation/length_mean_gm",
    "evaluation/length_std_gm",
    "evaluation/return_mean_gm",
    "evaluation/return_std_gm",
    "evaluation/return_vs_samples",
    "aug_traj/length",
    "aug_traj/return"
]

def load_event_logs(base_path, folder_path, metric_name):
    """Load event logs from a folder and extract metric values."""
    full_path = os.path.join(base_path, folder_path)
    event_files = [os.path.join(full_path, file) for file in os.listdir(full_path) 
                  if file.startswith("events.out.tfevents")]

    if not event_files:
        print(f"Warning: No event file found in {full_path}")
        return None

    # Sort event files by timestamp
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
        print(f"Warning: No data found for {metric_name} in {full_path}")
        return None

    # Convert to numpy arrays and sort by steps
    steps = np.array(steps)
    values = np.array(values)
    sort_idx = np.argsort(steps)
    steps = steps[sort_idx]
    values = values[sort_idx]

    return steps, values

def average_across_seeds(env_folders, metric_name):
    """Calculate average metric values across all seeds for an environment."""
    all_steps = []
    all_values = []
    
    for base_path, folder in env_folders:
        result = load_event_logs(base_path, folder, metric_name)
        if result is not None:
            steps, values = result
            all_steps.append(steps)
            all_values.append(values)
    
    if not all_steps:
        return None, None, None
    
    # Find common steps across all seeds
    common_steps = np.unique(np.concatenate(all_steps))
    
    # Interpolate values for each seed to match common steps
    interpolated_values = []
    for steps, values in zip(all_steps, all_values):
        interpolated = np.interp(common_steps, steps, values)
        interpolated_values.append(interpolated)
    
    # Calculate mean and std across seeds
    mean_values = np.mean(interpolated_values, axis=0)
    std_values = np.std(interpolated_values, axis=0)
    
    return common_steps, mean_values, std_values

def plot_metric(env_name, metric_name, steps, mean_values, std_values):
    """Create and save a plot for a specific metric and environment."""
    plt.figure(figsize=(10, 6))
    
    # Plot mean with confidence interval
    plt.plot(steps, mean_values, label='Mean', linewidth=2)
    plt.fill_between(steps, 
                    mean_values - std_values, 
                    mean_values + std_values, 
                    alpha=0.2, 
                    label='±1 std')
    
    plt.xlabel('Training Steps')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} - {env_name}')
    plt.legend()
    plt.grid(True)
    
    # Create directory structure with date and time
    now = datetime.now(tz=us_eastern)
    exp_folder = f"figures_ewa/{now.strftime('%Y.%m.%d')}/{env_name}"
    os.makedirs(exp_folder, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{exp_folder}/{now.strftime('%H%M%S')}_{metric_name.replace('/', '_')}.png")
    plt.close()

def main():
    # Process each environment
    for env_name, folders in envs.items():
        print(f"\nProcessing {env_name}...")
        
        # Process each metric
        for metric in metrics:
            print(f"  Processing {metric}...")
            
            # Calculate average across seeds
            steps, mean_values, std_values = average_across_seeds(folders, metric)
            
            if steps is not None:
                # Create and save plot
                plot_metric(env_name, metric, steps, mean_values, std_values)
            else:
                print(f"  Warning: No data available for {metric} in {env_name}")

if __name__ == "__main__":
    main()
