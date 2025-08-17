import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import pytz
import pandas as pd

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')

# Paths to experiment results
ewa_base_paths = [
    "/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.08.09"
]

odt_base_paths = [
    "/home/ubuntu/online_decision_transformer/online-dt-main/exp/2025.08.10"
]

# Define environments
envs = [
    "walker2d-medium-replay-v2",
    "hopper-medium-v2"
]

# Define metrics to compare (excluding EWA-specific metrics)
metrics = [
    "evaluation/return_mean_gm",
    "evaluation/return_std_gm", 
    "evaluation/return_vs_samples",
    "evaluation/length_mean_gm",
    "evaluation/length_std_gm",
    "aug_traj/return",
    "aug_traj/length"
]

# Global cache for event data
_event_cache = {}

def filter_folders_by_seeds(folders, env_name, start_seed=1, end_seed=3):
    """Filter experiment folders based on seed numbers."""
    import re
    filtered_folders = []
    for folder in folders:
        if env_name in folder:
            match = re.search(r'seed_(\d+)$', folder)
            if match:
                seed = int(match.group(1))
                if start_seed <= seed <= end_seed:
                    filtered_folders.append(folder)
    return filtered_folders

def collect_folders(base_paths, env_name, start_seed=1, end_seed=3):
    """Collect folders with seed filtering."""
    all_folders = []
    for base_path in base_paths:
        if os.path.exists(base_path):
            folders = [f for f in os.listdir(base_path) if env_name in f]
            filtered_folders = filter_folders_by_seeds(folders, env_name, start_seed, end_seed)
            all_folders.extend(filtered_folders)
    return all_folders

def load_event_logs(folder_path, metric_name):
    """Load event logs from a folder and extract metric values with caching."""
    global _event_cache
    
    cache_key = f"{folder_path}:{metric_name}"
    if cache_key in _event_cache:
        return _event_cache[cache_key]
    
    event_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                  if file.startswith("events.out.tfevents")]

    if not event_files:
        return None

    # Sort event files by timestamp and use the largest (most recent)
    event_files.sort()
    largest_file = max(event_files, key=lambda x: os.path.getsize(x))
    
    steps = []
    values = []
    
    try:
        for e in tf.compat.v1.train.summary_iterator(largest_file):
            for v in e.summary.value:
                if v.tag == metric_name:
                    steps.append(e.step)
                    values.append(v.simple_value)
                    if len(steps) > 1000:  # Limit data points
                        break
            if len(steps) > 1000:
                break
    except Exception as e:
        print(f"Warning: Error reading {largest_file}: {e}")
        return None

    if not steps or not values:
        return None

    # Convert to numpy arrays and sort by steps
    steps = np.array(steps)
    values = np.array(values)
    sort_idx = np.argsort(steps)
    steps = steps[sort_idx]
    values = values[sort_idx]
    
    result = (steps, values)
    _event_cache[cache_key] = result
    return result

def average_across_seeds(folders, metric_name, base_paths):
    """Calculate average metric values across all seeds for an environment."""
    all_steps = []
    all_values = []
    
    for folder in folders:
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

def get_final_metric_values(ewa_data, odt_data):
    """Extract average over all training steps, then averaged over seeds."""
    ewa_final = None
    odt_final = None
    
    if ewa_data is not None:
        steps, mean_values, std_values = ewa_data
        if len(mean_values) > 0:
            ewa_final = {
                'avg_value': float(np.mean(mean_values)),  # Average over all training steps
                'avg_std': float(np.mean(std_values)),     # Average std over all training steps
                'final_value': float(mean_values[-1]),     # Keep final value for reference
                'final_std': float(std_values[-1]),
                'max_value': float(np.max(mean_values)),
                'max_std': float(std_values[np.argmax(mean_values)])
            }
    
    if odt_data is not None:
        steps, mean_values, std_values = odt_data
        if len(mean_values) > 0:
            odt_final = {
                'avg_value': float(np.mean(mean_values)),  # Average over all training steps
                'avg_std': float(np.mean(std_values)),     # Average std over all training steps
                'final_value': float(mean_values[-1]),     # Keep final value for reference
                'final_std': float(std_values[-1]),
                'max_value': float(np.max(mean_values)),
                'max_std': float(std_values[np.argmax(mean_values)])
            }
    
    return ewa_final, odt_final

def generate_comparison_table():
    """Generate comparison table between EWA-ODT and ODT."""
    
    # Create output directory
    now = datetime.now(tz=us_eastern)
    output_dir = f"figures_comparison/{now.strftime('%Y.%m.%d')}/tables"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results storage
    results = {}
    
    for env_name in envs:
        print(f"\nProcessing {env_name}...")
        results[env_name] = {}
        
        # Collect folders for both EWA and ODT
        ewa_folders = collect_folders(ewa_base_paths, env_name, start_seed=1, end_seed=3)
        odt_folders = collect_folders(odt_base_paths, env_name, start_seed=1, end_seed=3)
        
        print(f"  EWA folders: {len(ewa_folders)}")
        print(f"  ODT folders: {len(odt_folders)}")
        
        for metric_name in metrics:
            print(f"    Processing {metric_name}...")
            
            # Process EWA data
            ewa_data = average_across_seeds(ewa_folders, metric_name, ewa_base_paths)
            
            # Process ODT data  
            odt_data = average_across_seeds(odt_folders, metric_name, odt_base_paths)
            
            # Extract final values
            ewa_final, odt_final = get_final_metric_values(ewa_data, odt_data)
            
            # Store results
            results[env_name][metric_name] = {
                'ewa': ewa_final,
                'odt': odt_final
            }
    
    # Generate comparison table
    generate_table_output(results, output_dir, now)
    
    return results

def generate_table_output(results, output_dir, timestamp):
    """Generate formatted table output."""
    
    # Create text file
    txt_file = f"{output_dir}/comparison_table_{timestamp.strftime('%H%M%S')}.txt"
    
    with open(txt_file, 'w') as f:
        f.write("EWA-ODT vs ODT Comparison Table\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} ET\n")
        f.write(f"Seeds: 1, 2, 3 (averaged)\n\n")
        
        for env_name in results:
            f.write(f"\n{env_name.upper()}\n")
            f.write("-" * len(env_name) + "\n\n")
            
            # Create table header
            f.write(f"{'Metric':<35} {'EWA-ODT Avg':<20} {'ODT Avg':<20} {'Improvement':<15}\n")
            f.write("-" * 90 + "\n")
            
            for metric_name in results[env_name]:
                ewa_data = results[env_name][metric_name]['ewa']
                odt_data = results[env_name][metric_name]['odt']
                
                if ewa_data and odt_data:
                    ewa_val = ewa_data['avg_value']  # Use average over all training steps
                    odt_val = odt_data['avg_value']  # Use average over all training steps
                    
                    # Calculate improvement
                    if odt_val != 0:
                        improvement = ((ewa_val - odt_val) / abs(odt_val)) * 100
                        improvement_str = f"{improvement:+.1f}%"
                    else:
                        improvement_str = "N/A"
                    
                    f.write(f"{metric_name:<35} {ewa_val:<20.4f} {odt_val:<20.4f} {improvement_str:<15}\n")
                else:
                    f.write(f"{metric_name:<35} {'N/A':<20} {'N/A':<20} {'N/A':<15}\n")
            
            f.write("\n")
    
    # Create CSV file
    csv_data = []
    for env_name in results:
        for metric_name in results[env_name]:
            ewa_data = results[env_name][metric_name]['ewa']
            odt_data = results[env_name][metric_name]['odt']
            
            row = {
                'Environment': env_name,
                'Metric': metric_name,
                'EWA_ODT_Avg': ewa_data['avg_value'] if ewa_data else None,
                'EWA_ODT_Avg_Std': ewa_data['avg_std'] if ewa_data else None,
                'EWA_ODT_Final': ewa_data['final_value'] if ewa_data else None,
                'EWA_ODT_Final_Std': ewa_data['final_std'] if ewa_data else None,
                'EWA_ODT_Max': ewa_data['max_value'] if ewa_data else None,
                'EWA_ODT_Max_Std': ewa_data['max_std'] if ewa_data else None,
                'ODT_Avg': odt_data['avg_value'] if odt_data else None,
                'ODT_Avg_Std': odt_data['avg_std'] if odt_data else None,
                'ODT_Final': odt_data['final_value'] if odt_data else None,
                'ODT_Final_Std': odt_data['final_std'] if odt_data else None,
                'ODT_Max': odt_data['max_value'] if odt_data else None,
                'ODT_Max_Std': odt_data['max_std'] if odt_data else None
            }
            
            # Calculate improvement percentages
            if ewa_data and odt_data:
                if odt_data['avg_value'] != 0:
                    row['Avg_Improvement_Pct'] = ((ewa_data['avg_value'] - odt_data['avg_value']) / abs(odt_data['avg_value'])) * 100
                else:
                    row['Avg_Improvement_Pct'] = None
                
                if odt_data['final_value'] != 0:
                    row['Final_Improvement_Pct'] = ((ewa_data['final_value'] - odt_data['final_value']) / abs(odt_data['final_value'])) * 100
                else:
                    row['Final_Improvement_Pct'] = None
                
                if odt_data['max_value'] != 0:
                    row['Max_Improvement_Pct'] = ((ewa_data['max_value'] - odt_data['max_value']) / abs(odt_data['max_value'])) * 100
                else:
                    row['Max_Improvement_Pct'] = None
            else:
                row['Avg_Improvement_Pct'] = None
                row['Final_Improvement_Pct'] = None
                row['Max_Improvement_Pct'] = None
            
            csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_file = f"{output_dir}/comparison_table_{timestamp.strftime('%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\nComparison table saved to:")
    print(f"  Text: {txt_file}")
    print(f"  CSV: {csv_file}")

def main():
    """Main function to generate comparison table."""
    print("Generating EWA-ODT vs ODT comparison table...")
    print("Metrics: evaluation and augmentation (excluding EWA-specific)")
    print("Seeds: 1, 2, 3 (averaged)")
    print("=" * 60)
    
    try:
        results = generate_comparison_table()
        print("\nComparison table generation completed successfully!")
        
        # Print summary
        print("\nSummary:")
        for env_name in results:
            print(f"\n{env_name}:")
            for metric_name in results[env_name]:
                ewa_data = results[env_name][metric_name]['ewa']
                odt_data = results[env_name][metric_name]['odt']
                
                if ewa_data and odt_data:
                    ewa_val = ewa_data['avg_value']  # Use average over all training steps
                    odt_val = odt_data['avg_value']  # Use average over all training steps
                    improvement = ((ewa_val - odt_val) / abs(odt_val)) * 100 if odt_val != 0 else 0
                    print(f"  {metric_name}: EWA-ODT {ewa_val:.4f} vs ODT {odt_val:.4f} ({improvement:+.1f}%)")
                else:
                    print(f"  {metric_name}: Missing data")
                    
    except Exception as e:
        print(f"Error generating comparison table: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
