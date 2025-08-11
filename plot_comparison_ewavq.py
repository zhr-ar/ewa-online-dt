import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import pytz
import pickle
import re

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')


ewa_base_paths_max_online_iters_10 = [
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.08.09"
]
ewa_base_paths = ewa_base_paths_max_online_iters_10

odt_base_paths_max_online_iters_10 = [
    "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.08.10"
]
odt_base_paths = odt_base_paths_max_online_iters_10


# # path to recent corrected A_t-1 EWA results with "max_online_iters": 30, "beta": 1.0, decay_ratio; 
# ewa_base_paths_max_online_iters_30_beta_1_0 = [
#     "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.05.11",
#     "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.05.12"
# ]
# ewa_base_paths = ewa_base_paths_max_online_iters_30_beta_1_0
# # path to ODT results with "max_online_iters": 30
# odt_base_paths_max_online_iters_30 = [
#     "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.21",
#     "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.22"
# ]
# odt_base_paths = odt_base_paths_max_online_iters_30


# # path to old experiments
# # path to EWA results with "max_online_iters": 30, "beta": 1.0, decay_ratio; which is promising
# ewa_base_paths_max_online_iters_30_beta_1_0 = [
#     "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.21"
# ]
# # path to EWA results with "max_online_iters": 30, "beta": 2.0, decay_ratio; which is NOT promising
# ewa_base_paths_max_online_iters_30_beta_2_0 = [
#     "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.22",
#     "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.23"
# ]

########################################################
# new experiments
########################################################

# # path to EWA results with "max_online_iters": 100, "beta": 1.0, decay_ratio
# ewa_base_paths = [
#     "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.25",
#     "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.26",
#     "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.27",
#     "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.28"
# ]

# # path to ODT results with "max_online_iters": 100
# odt_base_paths = [
#     "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.30",
#     "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.31",
#     "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.04.01"
# ]


########################################################

# Global cache for event data to avoid reloading the same files
_event_cache = {}

def filter_folders_by_seeds(folders, env_name, start_seed=1, end_seed=3):
    """
    Filter experiment folders based on seed numbers.
    
    Args:
        folders: List of folder names
        env_name: Name of the environment
        start_seed: Starting seed number (inclusive)
        end_seed: Ending seed number (inclusive)
        
    Returns:
        List of filtered folder names
    """
    filtered_folders = []
    for folder in folders:
        if env_name in folder:
            # Extract seed number from folder name
            match = re.search(r'seed_(\d+)$', folder)
            if match:
                seed = int(match.group(1))
                if start_seed <= seed <= end_seed:
                    filtered_folders.append(folder)
    return filtered_folders

# Define environments and their corresponding folders
envs = {
    # "halfcheetah-medium-expert-v2": [],
    "walker2d-medium-replay-v2": [],
    "hopper-medium-v2": []
}

# Function to collect folders with seed filtering
def collect_folders(base_paths, env_name, start_seed=1, end_seed=3):
    all_folders = []
    for base_path in base_paths:
        folders = [f for f in os.listdir(base_path) if env_name in f]
        filtered_folders = filter_folders_by_seeds(folders, env_name, start_seed, end_seed)
        all_folders.extend(filtered_folders)
    return all_folders

# Collect EWA folders with seed filtering
for env_name in envs.keys():
    ewa_folders = collect_folders(ewa_base_paths, env_name, start_seed=1, end_seed=3)  # Adjust seed range as needed
    envs[env_name].append({
        'type': 'ewa',
        'folders': ewa_folders
    })

# Collect ODT folders with seed filtering
for env_name in envs.keys():
    odt_folders = collect_folders(odt_base_paths, env_name, start_seed=1, end_seed=3)  # Adjust seed range as needed
    envs[env_name].append({
        'type': 'odt',
        'folders': odt_folders
    })

# Define metrics to plot
metrics = [
    "evaluation/return_mean_gm",
    "evaluation/return_std_gm",
    "evaluation/return_vs_samples",
    "evaluation/length_mean_gm",
    "evaluation/length_std_gm",
    "aug_traj/return",
    "aug_traj/length"
]
# EWA specific metrics are not compared with ODT
ewa_metrics = [
    'ewa/code_usage_layer_{layer_idx}', 
    'ewa/reward_layer_{layer_idx}',
    "ewa/attraction_layer_{layer_idx}"
]

# Group metrics by type
eval_metrics = [m for m in metrics if m.startswith("evaluation")]
aug_metrics = [m for m in metrics if m.startswith("aug_traj")]

def load_event_logs(folder_path, metric_name):
    """Load event logs from a folder and extract metric values with caching."""
    global _event_cache
    
    # Create a cache key for this folder and metric
    cache_key = f"{folder_path}:{metric_name}"
    
    # Check if we already have this data cached
    if cache_key in _event_cache:
        return _event_cache[cache_key]
    
    event_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                  if file.startswith("events.out.tfevents")]

    if not event_files:
        print(f"Warning: No event file found in {folder_path}")
        return None

    # Sort event files by timestamp
    event_files.sort()

    # Handle EWA-specific metrics that have layer indices
    if any(metric_name.startswith(prefix) for prefix in ["ewa/code_usage_layer_", "ewa/reward_layer_", "ewa/attraction_layer_"]):
        steps = []
        values = []
        
        # Process only the largest event file (most recent) to avoid processing multiple files
        largest_file = max(event_files, key=lambda x: os.path.getsize(x))
        
        try:
            for e in tf.compat.v1.train.summary_iterator(largest_file):
                for v in e.summary.value:
                    if v.tag == metric_name:
                        steps.append(e.step)
                        values.append(v.simple_value)
                        # Early exit if we have enough data points
                        if len(steps) > 1000:  # Limit to reasonable number of points
                            break
                if len(steps) > 1000:
                    break
        except Exception as e:
            print(f"Warning: Error reading {largest_file}: {e}")
            return None

        if not steps or not values:
            print(f"Warning: No data found for {metric_name} in {folder_path}")
            return None

        # Convert to numpy arrays and sort by steps
        steps = np.array(steps)
        values = np.array(values)
        sort_idx = np.argsort(steps)
        steps = steps[sort_idx]
        values = values[sort_idx]
        
        # Cache the result
        result = (steps, values)
        _event_cache[cache_key] = result
        return result
    else:
        # Handle regular metrics as before
        steps = []
        values = []
        
        # Process only the largest event file for regular metrics too
        largest_file = max(event_files, key=lambda x: os.path.getsize(x))
        
        try:
            for e in tf.compat.v1.train.summary_iterator(largest_file):
                for v in e.summary.value:
                    if v.tag == metric_name:
                        steps.append(e.step)
                        values.append(v.simple_value)
                        # Early exit if we have enough data points
                        if len(steps) > 1000:  # Limit to reasonable number of points
                            break
                if len(steps) > 1000:
                    break
        except Exception as e:
            print(f"Warning: Error reading {largest_file}: {e}")
            return None

        if not steps or not values:
            print(f"Warning: No data found for {metric_name} in {folder_path}")
            return None

        # Convert to numpy arrays and sort by steps
        steps = np.array(steps)
        values = np.array(values)
        sort_idx = np.argsort(steps)
        steps = steps[sort_idx]
        values = values[sort_idx]

        # Cache the result
        result = (steps, values)
        _event_cache[cache_key] = result
        return result

def average_across_seeds(folders, metric_name, base_paths):
    """Calculate average metric values across all seeds for an environment."""
    all_steps = []
    all_values = []
    all_steps_ewa = []
    all_values_ewa = []
    for folder in folders:
        # Try all base paths
        for base_path in base_paths:
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                result = load_event_logs(folder_path, metric_name)
                if result is not None:
                    if metric_name in ewa_metrics:
                        print(f"result: {result}")
                        steps_ewa, values_ewa = result
                        all_steps_ewa.append(steps_ewa)
                        all_values_ewa.append(values_ewa)
                    else:
                        steps, values = result
                        all_steps.append(steps)
                        all_values.append(values)
    
    if not all_steps:
        return None, None, None
    
    # Find common steps across all seeds
    common_steps = np.unique(np.concatenate(all_steps))
    
    # Interpolate values for each seed to match common steps
    interpolated_values = []
    interpolated_values_ewa = []
    
    for steps, values in zip(all_steps, all_values):
        interpolated = np.interp(common_steps, steps, values)
        interpolated_values.append(interpolated)
    
    # Calculate mean and std across seeds
    mean_values = np.mean(interpolated_values, axis=0)
    std_values = np.std(interpolated_values, axis=0)    
    
    if metric_name in ewa_metrics:
        common_steps_ewa = np.unique(np.concatenate(all_steps_ewa))
        # Interpolate decay ratios
        for steps_ewa, values_ewa in zip(all_steps_ewa, all_values_ewa):
            interpolated_ewa = np.interp(common_steps, steps_ewa, values_ewa)
            interpolated_values_ewa.append(interpolated_ewa)
        
        # Calculate mean decay ratio
        mean_values_ewa = np.mean(interpolated_values_ewa, axis=0)
        std_values_ewa = np.std(interpolated_values_ewa, axis=0)
        
        return common_steps_ewa, mean_values_ewa, std_values_ewa
    
    return common_steps, mean_values, std_values

def plot_comparison(env_name, metric_name, ewa_data, odt_data):
    """Create and save a comparison plot for a specific metric and environment."""
    plt.figure(figsize=(12, 7))
    
    # For EWA-specific metrics 
    if metric_name in ewa_metrics:
        if ewa_data is not None:
            steps, mean_values, std_values = ewa_data
            plt.plot(steps, mean_values, label='EWA', linewidth=2, color='blue')
            plt.fill_between(steps, 
                           mean_values - std_values, 
                           mean_values + std_values, 
                           alpha=0.2, 
                           color='blue',
                           label='EWA ±1 std')
    else:
        # For shared metrics, plot both EWA and ODT data
        # Plot EWA data
        if ewa_data is not None:
            steps, mean_values, std_values = ewa_data
            plt.plot(steps, mean_values, label='EWA', linewidth=2, color='blue')
            plt.fill_between(steps, 
                           mean_values - std_values, 
                           mean_values + std_values, 
                           alpha=0.2, 
                           color='blue',
                           label='EWA ±1 std')
        
        # Plot ODT data
        if odt_data is not None:
            steps, mean_values, std_values = odt_data
            plt.plot(steps, mean_values, label='ODT', linewidth=2, color='red')
            
            # Ensure std_values are not too small to be visible
            min_visible_std = np.max(std_values) * 0.01  # 1% of max std
            std_values = np.maximum(std_values, min_visible_std)
            
            plt.fill_between(steps, 
                           mean_values - std_values, 
                           mean_values + std_values, 
                           alpha=0.2, 
                           color='red',
                           label='ODT ±1 std')
    
    plt.xlabel('Training Steps')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} - {env_name}')
    plt.legend()
    plt.grid(True)
    
    # Create directory structure with date and time
    now = datetime.now(tz=us_eastern)
    exp_folder = f"figures_comparison/{now.strftime('%Y.%m.%d')}/{env_name}"
    os.makedirs(exp_folder, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{exp_folder}/{now.strftime('%H%M%S')}_{metric_name.replace('/', '_')}.png")
    plt.close()

def plot_combined_metrics(env_name, metric_data, metric_type="evaluation"):
    """Create a combined plot with subplots for all metrics of a given type."""
    if metric_type == "evaluation":
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        metrics_to_plot = eval_metrics
        title = f"Evaluation Metrics - {env_name}"
    else:  # augmentation
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes = axes.flatten()
        metrics_to_plot = aug_metrics
        title = f"Augmentation Metrics - {env_name}"
    
    fig.suptitle(title, fontsize=16, y=1.02)
    
    for ax, metric in zip(axes, metrics_to_plot):
        ewa_data = metric_data.get(("ewa", metric))
        odt_data = metric_data.get(("odt", metric))
        
        # Plot EWA data
        if ewa_data is not None:
            steps, mean_values, std_values = ewa_data
            ax.plot(steps, mean_values, label='EWA', linewidth=2, color='blue')
            ax.fill_between(steps, 
                          mean_values - std_values, 
                          mean_values + std_values, 
                          alpha=0.2, 
                          color='blue',
                          label='EWA ±1 std')
        
        # Plot ODT data
        if odt_data is not None:
            steps, mean_values, std_values = odt_data
            ax.plot(steps, mean_values, label='ODT', linewidth=2, color='red')
            
            # Ensure std_values are not too small to be visible
            min_visible_std = np.max(std_values) * 0.01
            std_values = np.maximum(std_values, min_visible_std)
            
            ax.fill_between(steps, 
                          mean_values - std_values, 
                          mean_values + std_values, 
                          alpha=0.2, 
                          color='red',
                          label='ODT ±1 std')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.split('/')[-1].replace('_', ' ').title())
        ax.legend()
        ax.grid(True)
    
    # Remove any extra subplots for evaluation metrics
    if metric_type == "evaluation" and len(metrics_to_plot) < 6:
        for i in range(len(metrics_to_plot), 6):
            fig.delaxes(axes[i])
    
    # Adjust layout
    plt.tight_layout()
    
    # Create directory structure with date and time
    now = datetime.now(tz=us_eastern)
    exp_folder = f"figures_comparison/{now.strftime('%Y.%m.%d')}/{env_name}"
    os.makedirs(exp_folder, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{exp_folder}/{now.strftime('%H%M%S')}_{metric_type}_combined.png", 
                bbox_inches='tight', dpi=300)
    plt.close()

def get_dataset_returns(env_name):
    """Load and analyze the dataset to get min/max returns."""
    dataset_path = f"/home/ubuntu/online_decision_transformer/ewa-online-dt-main/data/{env_name}.pkl"
    try:
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
        
        # Calculate returns for each trajectory
        returns = []
        for path in trajectories:
            returns.append(np.sum(path["rewards"]))
        returns = np.array(returns)
        
        print(f"\nDataset statistics for {env_name}:")
        print(f"Number of trajectories: {len(returns)}")
        print(f"Min return: {np.min(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}")
        print(f"Mean return: {np.mean(returns):.2f}")
        print(f"Std return: {np.std(returns):.2f}")
        
        return {"min": float(np.min(returns)), "max": float(np.max(returns))}
    except Exception as e:
        print(f"Error loading dataset for {env_name}: {e}")
        return None

def calculate_normalized_returns():
    """Calculate and save normalized returns for both EWA and ODT experiments."""
    # Get actual min/max returns from datasets
    env_returns = {}
    for env_name in envs.keys():
        returns = get_dataset_returns(env_name)
        if returns:
            env_returns[env_name] = returns
        else:
            print(f"Skipping {env_name} due to missing dataset")
            continue
    
    if not env_returns:
        print("No datasets found. Cannot calculate normalized returns.")
        return
    
    # Create output directory
    output_dir = "normalized_returns"
    os.makedirs(output_dir, exist_ok=True)
    
    # Current timestamp
    now = datetime.now(tz=us_eastern)
    output_file = f"{output_dir}/normalized_returns_{now.strftime('%Y.%m.%d_%H%M%S')}.txt"
    
    def normalize_return(value, env_min, env_max):
        """Normalize return to [0, 100] range."""
        return 100 * (value - env_min) / (env_max - env_min)
    
    with open(output_file, "w") as f:
        f.write("Normalized Returns (0-100 scale)\n")
        f.write("=" * 50 + "\n\n")
        
        for env_name, env_info in env_returns.items():
            print(f"\nProcessing {env_name}...")
            f.write(f"Environment: {env_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Dataset min return: {env_info['min']:.2f}\n")
            f.write(f"Dataset max return: {env_info['max']:.2f}\n\n")
            
            # Process EWA data
            ewa_returns = []
            for base_path in ewa_base_paths:
                ewa_folders = [folder for folder in os.listdir(base_path) if env_name in folder]
                for folder in ewa_folders:
                    folder_path = os.path.join(base_path, folder)
                    result = load_event_logs(folder_path, "evaluation/return_mean_gm")
                    if result is not None:
                        steps, values = result
                        if len(values) > 0:
                            final_return = float(values[-1])  # Convert to float
                            normalized_return = normalize_return(final_return, env_info["min"], env_info["max"])
                            ewa_returns.append(normalized_return)
                            f.write(f"Raw return (EWA): {final_return:.2f}, Normalized: {normalized_return:.2f}\n")
            
            # Process ODT data
            odt_returns = []
            for base_path in odt_base_paths:
                odt_folders = [folder for folder in os.listdir(base_path) if env_name in folder]
                for folder in odt_folders:
                    folder_path = os.path.join(base_path, folder)
                    result = load_event_logs(folder_path, "evaluation/return_mean_gm")
                    if result is not None:
                        steps, values = result
                        if len(values) > 0:
                            final_return = float(values[-1])  # Convert to float
                            normalized_return = normalize_return(final_return, env_info["min"], env_info["max"])
                            odt_returns.append(normalized_return)
                            f.write(f"Raw return (ODT): {final_return:.2f}, Normalized: {normalized_return:.2f}\n")
            
            f.write("\n")
            # Write summary statistics
            if ewa_returns:
                ewa_mean = np.mean(ewa_returns)
                ewa_std = np.std(ewa_returns)
                f.write(f"EWA Summary (n={len(ewa_returns)}): {ewa_mean:.2f} ± {ewa_std:.2f}\n")
            else:
                f.write("EWA: No data found\n")
            
            if odt_returns:
                odt_mean = np.mean(odt_returns)
                odt_std = np.std(odt_returns)
                f.write(f"ODT Summary (n={len(odt_returns)}): {odt_mean:.2f} ± {odt_std:.2f}\n")
            else:
                f.write("ODT: No data found\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"\nNormalized returns have been saved to: {output_file}")

def save_iteration_returns(f, env_name, env_info, metric_data):
    """Save average normalized and raw returns for each iteration."""
    f.write("\nIteration-wise Returns\n")
    f.write("-" * 30 + "\n")
    
    # Get the data for return_mean_gm metric
    ewa_data = metric_data.get(("ewa", "evaluation/return_mean_gm"))
    odt_data = metric_data.get(("odt", "evaluation/return_mean_gm"))
    
    if ewa_data is not None and odt_data is not None:
        steps, ewa_means, _ = ewa_data
        _, odt_means, _ = odt_data
        
        f.write(f"\nStep,EWA Raw,EWA Normalized,ODT Raw,ODT Normalized\n")
        
        def normalize_return(value, env_min, env_max):
            return 100 * (value - env_min) / (env_max - env_min)
        
        for i, step in enumerate(steps):
            ewa_raw = ewa_means[i]
            odt_raw = odt_means[i]
            ewa_norm = normalize_return(ewa_raw, env_info["min"], env_info["max"])
            odt_norm = normalize_return(odt_raw, env_info["min"], env_info["max"])
            
            f.write(f"{step} || {ewa_raw:.2f} | {ewa_norm:.2f} || {odt_raw:.2f} | {odt_norm:.2f}\n")
    
    f.write("\n" + "=" * 50 + "\n")

def plot_ewa_metrics_combined(env_name, ewa_data_dict):
    """Create a combined plot showing all EWA metrics for all layers."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'EWA Metrics - {env_name}', fontsize=16, y=1.02)
    
    # Define metric types and their positions
    metric_types = ['reward', 'code_usage', 'attraction']
    
    for row, metric_type in enumerate(metric_types):
        for col in range(4):  # 4 layers
            ax = axes[row, col]
            metric_name = f"ewa/{metric_type}_layer_{col}"
            
            if metric_name in ewa_data_dict:
                steps, mean_values, std_values = ewa_data_dict[metric_name]
                
                # Plot mean values with standard deviation
                ax.plot(steps, mean_values, linewidth=2, label='Mean')
                ax.fill_between(steps, 
                              mean_values - std_values, 
                              mean_values + std_values, 
                              alpha=0.2)
                
                ax.set_xlabel('Training Steps')
                ax.set_ylabel(f'{metric_type.replace("_", " ").title()}')
                ax.set_title(f'Layer {col}')
                ax.grid(True)
    
    plt.tight_layout()
    
    # Create directory structure with date and time
    now = datetime.now(tz=us_eastern)
    exp_folder = f"figures_comparison/{now.strftime('%Y.%m.%d')}/{env_name}"
    os.makedirs(exp_folder, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{exp_folder}/{now.strftime('%H%M%S')}_ewa_metrics_combined.png", 
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Create output directory and file at the start
    now = datetime.now(tz=us_eastern)
    output_dir = f"figures_comparison/{now.strftime('%Y.%m.%d')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each environment
    for env_name in envs:
        print(f"\nProcessing {env_name}...")
        
        # Get folders for both EWA and ODT
        ewa_folders = collect_folders(ewa_base_paths, env_name)
        odt_folders = collect_folders(odt_base_paths, env_name)
        
        # Dictionary to store all EWA metrics data
        ewa_metrics_data = {}
        
        # Process EWA-specific metrics for each layer
        print(f"  Processing EWA metrics for {len(ewa_folders)} folders...")
        for metric_base in ["ewa/code_usage_layer_", "ewa/reward_layer_", "ewa/attraction_layer_"]:
            for layer_idx in range(4):  # 4 layers
                metric_name = f"{metric_base}{layer_idx}"
                print(f"    Processing {metric_name}...", end=" ")
                
                # Process EWA data
                ewa_data = average_across_seeds(ewa_folders, metric_name, ewa_base_paths)
                if ewa_data is not None:
                    ewa_metrics_data[metric_name] = ewa_data
                    print("✓")
                else:
                    print("✗ (no data)")
        
        # Create combined plot for all EWA metrics
        if ewa_metrics_data:
            print("\n  Creating combined EWA metrics plot...")
            plot_ewa_metrics_combined(env_name, ewa_metrics_data)
        
        # Dictionary to store all regular metrics data
        all_metric_data = {}
        
        # Process regular metrics
        for metric_name in metrics:
            print(f"\n  Processing {metric_name}...")
            
            # Process EWA data
            print("    Processing EWA data...")
            ewa_data = average_across_seeds(ewa_folders, metric_name, ewa_base_paths)
            if ewa_data is not None:
                all_metric_data[("ewa", metric_name)] = ewa_data
            
            # Process ODT data
            print("    Processing ODT data...")
            odt_data = average_across_seeds(odt_folders, metric_name, odt_base_paths)
            if odt_data is not None:
                all_metric_data[("odt", metric_name)] = odt_data
        
        # Create combined plots for regular metrics
        if all_metric_data:
            print("\n  Creating combined evaluation plot...")
            plot_combined_metrics(env_name, all_metric_data, "evaluation")
            print("  Creating combined augmentation plot...")
            plot_combined_metrics(env_name, all_metric_data, "augmentation")

if __name__ == "__main__":
    main() 