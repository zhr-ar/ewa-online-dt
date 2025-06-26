import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import pytz
import pickle

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')

"""
EWA Experiment Settings Documentation
===================================

The EWA results plotted in this file were collected using the following settings from ewa_decay.py:

1. Number of Revisits Calculation:
   n_visits = (max_online_iters * num_updates_per_online_iter * batch_size) // (replay_size * trajectory_length)
   This determines when the delta value starts decaying from initial_delta to final_delta.

2. Decay Step Calculation:
   decay_step = max_online_iters / n_visits
   This determines the rate at which delta decays.

These settings were used consistently across all EWA experiments whose results are plotted here.
"""

# path to old experiments
# path to EWA results with "max_online_iters": 30, "beta": 1.0, decay_ratio; which is promising
ewa_base_paths_max_online_iters_30_beta_1_0 = [
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.21"
]
# path to EWA results with "max_online_iters": 30, "beta": 2.0, decay_ratio; which is NOT promising
ewa_base_paths_max_online_iters_30_beta_2_0 = [
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.22",
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.23"
]

# path to ODT results with "max_online_iters": 30
odt_base_paths_max_online_iters_30 = [
    "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.21",
    "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.22"
]

########################################################
# new experiments
########################################################

# path to EWA results with "max_online_iters": 100, "beta": 1.0, decay_ratio
ewa_base_paths = [
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.25",
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.26",
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.27",
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.03.28"
]

# path to ODT results with "max_online_iters": 100
odt_base_paths = [
    "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.30",
    "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.31",
    "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.04.01"
]

# Define environments and their corresponding folders
envs = {
    "halfcheetah-medium-expert-v2": [],
    "walker2d-medium-replay-v2": [],
    "hopper-medium-v2": []
}

# Collect EWA folders
for env_name in envs.keys():
    ewa_folders = []
    for base_path in ewa_base_paths:
        ewa_folders.extend([f for f in os.listdir(base_path) if env_name in f])
    envs[env_name].append({
        'type': 'ewa',
        'folders': ewa_folders
    })

# Collect ODT folders from both base paths
for env_name in envs.keys():
    odt_folders = []
    for base_path in odt_base_paths:
        odt_folders.extend([f for f in os.listdir(base_path) if env_name in f])
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
    "aug_traj/length",
    "delta"
]

# Group metrics by type
eval_metrics = [m for m in metrics if m.startswith("evaluation")]
aug_metrics = [m for m in metrics if m.startswith("aug_traj")]

def load_event_logs(folder_path, metric_name):
    """Load event logs from a folder and extract metric values."""
    event_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                  if file.startswith("events.out.tfevents")]

    if not event_files:
        print(f"Warning: No event file found in {folder_path}")
        return None

    # Sort event files by timestamp
    event_files.sort()

    steps = []
    values = []
    decay_ratios = []  # New list for decay ratios
    
    for event_file in event_files:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                if v.tag == metric_name:
                    steps.append(e.step)
                    values.append(v.simple_value)
                # If this is delta data, also look for decay ratio
                elif metric_name == "delta" and v.tag == "decay_ratio":
                    decay_ratios.append(v.simple_value)

    if not steps or not values:
        print(f"Warning: No data found for {metric_name} in {folder_path}")
        return None

    # Convert to numpy arrays and sort by steps
    steps = np.array(steps)
    values = np.array(values)
    sort_idx = np.argsort(steps)
    steps = steps[sort_idx]
    values = values[sort_idx]

    # If this is delta data and we have decay ratios, return both
    if metric_name == "delta" and decay_ratios:
        decay_ratios = np.array(decay_ratios)[sort_idx]
        return steps, values, decay_ratios

    return steps, values

def average_across_seeds(folders, metric_name, base_paths):
    """Calculate average metric values across all seeds for an environment."""
    all_steps = []
    all_values = []
    all_decay_ratios = []  # New list for decay ratios
    
    for folder in folders:
        # Try all base paths
        for base_path in base_paths:
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                result = load_event_logs(folder_path, metric_name)
                if result is not None:
                    if metric_name == "delta" and len(result) == 3:
                        steps, values, decay_ratios = result
                        all_steps.append(steps)
                        all_values.append(values)
                        all_decay_ratios.append(decay_ratios)
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
    interpolated_decay_ratios = []  # New list for interpolated decay ratios
    
    for steps, values in zip(all_steps, all_values):
        interpolated = np.interp(common_steps, steps, values)
        interpolated_values.append(interpolated)
    
    # Calculate mean and std across seeds
    mean_values = np.mean(interpolated_values, axis=0)
    std_values = np.std(interpolated_values, axis=0)
    
    # If this is delta data and we have decay ratios, return both
    if metric_name == "delta" and all_decay_ratios:
        # Interpolate decay ratios
        for steps, decay_values in zip(all_steps, all_decay_ratios):
            interpolated = np.interp(common_steps, steps, decay_values)
            interpolated_decay_ratios.append(interpolated)
        
        # Calculate mean decay ratio
        mean_decay_ratio = np.mean(interpolated_decay_ratios, axis=0)
        
        return common_steps, mean_values, mean_decay_ratio
    
    return common_steps, mean_values, std_values

def plot_comparison(env_name, metric_name, ewa_data, odt_data):
    """Create and save a comparison plot for a specific metric and environment."""
    plt.figure(figsize=(12, 7))
    
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

def plot_delta_and_decay_ratio(env_name, ewa_data):
    """Create and save a plot showing both delta and decay ratio over training steps."""
    plt.figure(figsize=(12, 7))
    
    # Plot delta and decay ratio
    if ewa_data is not None:
        steps, delta_values, decay_ratios = ewa_data
        
        # Create two y-axes
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot delta on left y-axis
        line1, = ax1.plot(steps, delta_values, label='Delta', linewidth=2, color='blue')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Delta Value', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot decay ratio on right y-axis
        line2, = ax2.plot(steps, decay_ratios, label='Decay Ratio', linewidth=2, color='red')
        ax2.set_ylabel('Decay Ratio', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title(f'Delta and Decay Ratio Over Training - {env_name}')
        plt.grid(True)
    
    # Create directory structure with date and time
    now = datetime.now(tz=us_eastern)
    exp_folder = f"figures_comparison/{now.strftime('%Y.%m.%d')}/{env_name}"
    os.makedirs(exp_folder, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{exp_folder}/{now.strftime('%H%M%S')}_delta_and_decay_ratio.png")
    plt.close()

def main():
    # Create output directory and file at the start
    output_dir = "normalized_returns"
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now(tz=us_eastern)
    output_file = f"{output_dir}/normalized_returns_{now.strftime('%Y.%m.%d_%H%M%S')}.txt"

    # Dictionary to store all environment data
    all_env_data = {}
    
    # First pass: collect all data
    for env_name, data_sources in envs.items():
        print(f"\nProcessing {env_name}...")
        
        # Dictionary to store all metric data for this environment
        all_metric_data = {}
        
        # Process each metric
        for metric in metrics:
            print(f"\n  Processing {metric}...")
            
            # Get EWA data
            for source in data_sources:
                if source['type'] == 'ewa':
                    print("    Processing EWA data...")
                    ewa_data = average_across_seeds(source['folders'], metric, ewa_base_paths)
                    all_metric_data[("ewa", metric)] = ewa_data
                    break
            
            # Get ODT data
            for source in data_sources:
                if source['type'] == 'odt':
                    print("    Processing ODT data...")
                    odt_data = average_across_seeds(source['folders'], metric, odt_base_paths)
                    all_metric_data[("odt", metric)] = odt_data
                    break
            
            # Create individual comparison plot
            plot_comparison(env_name, metric, ewa_data, odt_data)
        
        # Create combined plots
        print("\n  Creating combined evaluation plot...")
        plot_combined_metrics(env_name, all_metric_data, "evaluation")
        print("  Creating combined augmentation plot...")
        plot_combined_metrics(env_name, all_metric_data, "augmentation")
        
        # Plot delta and decay ratio if available
        if ("ewa", "delta") in all_metric_data:
            print("  Creating delta and decay ratio plot...")
            plot_delta_and_decay_ratio(env_name, all_metric_data[("ewa", "delta")])
        
        # Store the metric data and get dataset returns
        dataset_returns = get_dataset_returns(env_name)
        if dataset_returns:
            all_env_data[env_name] = {
                'metric_data': all_metric_data,
                'dataset_returns': dataset_returns
            }
    
    # Second pass: write everything to the same file
    with open(output_file, "w") as f:
        # First write iteration-wise returns for all environments
        f.write("ITERATION-WISE RETURNS FOR ALL ENVIRONMENTS\n")
        f.write("=" * 50 + "\n\n")
        
        for env_name, env_data in all_env_data.items():
            f.write(f"\nIteration-wise returns for {env_name}\n")
            save_iteration_returns(f, env_name, env_data['dataset_returns'], env_data['metric_data'])
        
        # Then write the final normalized returns summary
        f.write("\n\nFINAL NORMALIZED RETURNS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for env_name, env_data in all_env_data.items():
            dataset_returns = env_data['dataset_returns']
            f.write(f"Environment: {env_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Dataset min return: {dataset_returns['min']:.2f}\n")
            f.write(f"Dataset max return: {dataset_returns['max']:.2f}\n\n")
            
            def normalize_return(value, env_min, env_max):
                return 100 * (value - env_min) / (env_max - env_min)
            
            # Process EWA data
            ewa_seed_means = []
            for base_path in ewa_base_paths:
                ewa_folders = [folder for folder in os.listdir(base_path) if env_name in folder]
                for folder in ewa_folders:
                    folder_path = os.path.join(base_path, folder)
                    result = load_event_logs(folder_path, "evaluation/return_mean_gm")
                    if result is not None:
                        steps, values = result
                        # Calculate average normalized return for this seed
                        normalized_values = [normalize_return(val, dataset_returns["min"], dataset_returns["max"]) for val in values]
                        seed_mean = np.mean(normalized_values)
                        ewa_seed_means.append(seed_mean)
                        f.write(f"Seed average (EWA): {seed_mean:.2f}\n")
            
            # Process ODT data
            odt_seed_means = []
            for base_path in odt_base_paths:
                odt_folders = [folder for folder in os.listdir(base_path) if env_name in folder]
                for folder in odt_folders:
                    folder_path = os.path.join(base_path, folder)
                    result = load_event_logs(folder_path, "evaluation/return_mean_gm")
                    if result is not None:
                        steps, values = result
                        # Calculate average normalized return for this seed
                        normalized_values = [normalize_return(val, dataset_returns["min"], dataset_returns["max"]) for val in values]
                        seed_mean = np.mean(normalized_values)
                        odt_seed_means.append(seed_mean)
                        f.write(f"Seed average (ODT): {seed_mean:.2f}\n")
            
            f.write("\n")
            # Write summary statistics across seeds
            if ewa_seed_means:
                ewa_mean = np.mean(ewa_seed_means)
                ewa_std = np.std(ewa_seed_means)
                f.write(f"EWA Summary (n={len(ewa_seed_means)} seeds): {ewa_mean:.2f} ± {ewa_std:.2f}\n")
            else:
                f.write("EWA: No data found\n")
            
            if odt_seed_means:
                odt_mean = np.mean(odt_seed_means)
                odt_std = np.std(odt_seed_means)
                f.write(f"ODT Summary (n={len(odt_seed_means)} seeds): {odt_mean:.2f} ± {odt_std:.2f}\n")
            else:
                f.write("ODT: No data found\n")
            
            f.write("\n" + "=" * 50 + "\n")
    
    print(f"\nAll results have been saved to: {output_file}")

if __name__ == "__main__":
    main() 