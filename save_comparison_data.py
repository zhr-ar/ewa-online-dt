import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import pytz

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')

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

# # path to ODT results with "max_online_iters": 30
# odt_base_paths_max_online_iters_30 = [
#     "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.21",
#     "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.03.22"
# ]

########################################################
# new experiments
########################################################

ewa_base_paths_max_online_iters_10 = [
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.08.09"
]
ewa_base_paths = ewa_base_paths_max_online_iters_10

odt_base_paths_max_online_iters_10 = [
    "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.08.10"
]
odt_base_paths = odt_base_paths_max_online_iters_10

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

# Define environments and their corresponding folders
envs = {
    # "halfcheetah-medium-expert-v2": [],
    "walker2d-medium-replay-v2": [],
    "hopper-medium-v2": []
}

# Collect EWA folders from all base paths
for env_name in envs.keys():
    ewa_folders = []
    for base_path in ewa_base_paths:
        ewa_folders.extend([f for f in os.listdir(base_path) if env_name in f])
    envs[env_name].append({
        'type': 'ewa',
        'folders': ewa_folders
    })

# Collect ODT folders from all base paths
for env_name in envs.keys():
    odt_folders = []
    for base_path in odt_base_paths:  # Use odt_base_paths list instead of hardcoded paths
        odt_folders.extend([f for f in os.listdir(base_path) if env_name in f])
    envs[env_name].append({
        'type': 'odt',
        'folders': odt_folders
    })

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
    for event_file in event_files:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                if v.tag == metric_name:
                    steps.append(e.step)
                    values.append(v.simple_value)

    if not steps or not values:
        print(f"Warning: No data found for {metric_name} in {folder_path}")
        return None

    # Convert to numpy arrays and sort by steps
    steps = np.array(steps)
    values = np.array(values)
    sort_idx = np.argsort(steps)
    steps = steps[sort_idx]
    values = values[sort_idx]

    return steps, values

def average_across_seeds(folders, metric_name, base_paths):
    """Calculate average metric values across all seeds for an environment."""
    all_steps = []
    all_values = []
    
    for folder in folders:
        # Try all base paths
        for base_path in base_paths:
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                result = load_event_logs(folder_path, metric_name)
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

def save_comparison_data(env_name, metric_name, ewa_data, odt_data):
    """Save comparison data in a compressed format."""
    # Create directory structure
    now = datetime.now(tz=us_eastern)
    save_dir = f"figures_comparison/data_comparison/{now.strftime('%Y.%m.%d')}/{env_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data dictionary
    data = {
        'metric': metric_name,
        'timestamp': now.strftime('%Y.%m.%d/%H%M%S'),
        'ewa': None,
        'odt': None
    }
    
    # Add EWA data if available
    if ewa_data is not None:
        steps, mean_values, std_values = ewa_data
        data['ewa'] = {
            'steps': steps,
            'mean': mean_values,
            'std': std_values
        }
    
    # Add ODT data if available
    if odt_data is not None:
        steps, mean_values, std_values = odt_data
        data['odt'] = {
            'steps': steps,
            'mean': mean_values,
            'std': std_values
        }
    
    # Save data in compressed format
    save_path = f"{save_dir}/{now.strftime('%H%M%S')}_{metric_name.replace('/', '_')}.npz"
    np.savez_compressed(save_path, **data)
    
    # Print file size
    file_size = os.path.getsize(save_path) / 1024  # Convert to KB
    print(f"Saved data to {save_path} ({file_size:.2f} KB)")

def main():
    # Process each environment
    for env_name, data_sources in envs.items():
        print(f"\nProcessing {env_name}...")
        
        # Process each metric
        for metric in metrics:
            print(f"\n  Processing {metric}...")
            
            # Get EWA data
            ewa_data = None
            for source in data_sources:
                if source['type'] == 'ewa':
                    print("    Processing EWA data...")
                    ewa_data = average_across_seeds(source['folders'], metric, ewa_base_paths)
                    break
            
            # Get ODT data
            odt_data = None
            for source in data_sources:
                if source['type'] == 'odt':
                    print("    Processing ODT data...")
                    odt_data = average_across_seeds(source['folders'], metric, odt_base_paths)
                    break
            
            # Save comparison data
            save_comparison_data(env_name, metric, ewa_data, odt_data)

if __name__ == "__main__":
    main() 