import os
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')

# Path to the data comparison folder
data_comparison_base = "figures_comparison/data_comparison/2025.08.12"

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

def load_npz_data(env_name, metric_name):
    """Load data from .npz files for a specific environment and metric."""
    env_dir = os.path.join(data_comparison_base, env_name)
    if not os.path.exists(env_dir):
        print(f"Warning: Directory {env_dir} does not exist")
        return None, None
    
    # Find the .npz file for this metric
    metric_filename = metric_name.replace('/', '_')
    npz_files = [f for f in os.listdir(env_dir) if f.endswith('.npz') and metric_filename in f]
    
    if not npz_files:
        print(f"Warning: No .npz file found for {metric_name} in {env_name}")
        return None, None
    
    # Use the most recent file if multiple exist
    npz_file = sorted(npz_files)[-1]
    npz_path = os.path.join(env_dir, npz_file)
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        ewa_data = None
        odt_data = None
        
        if 'ewa' in data.files and data['ewa'] is not None:
            ewa_data = {
                'steps': data['ewa'].item()['steps'],
                'mean_values': data['ewa'].item()['mean'],
                'std_values': data['ewa'].item()['std']
            }
        
        if 'odt' in data.files and data['odt'] is not None:
            odt_data = {
                'steps': data['odt'].item()['steps'],
                'mean_values': data['odt'].item()['mean'],
                'std_values': data['odt'].item()['std']
            }
        
        print(f"    Loaded {npz_file} ({os.path.getsize(npz_path)/1024:.1f} KB)")
        return ewa_data, odt_data
        
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None, None

def get_metric_values(ewa_data, odt_data):
    """Extract average over all training steps from the loaded data."""
    ewa_final = None
    odt_final = None
    
    if ewa_data is not None:
        mean_values = ewa_data['mean_values']
        std_values = ewa_data['std_values']
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
        mean_values = odt_data['mean_values']
        std_values = odt_data['std_values']
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
    """Generate comparison table between EWA-ODT and ODT from .npz files."""
    
    # Create output directory
    now = datetime.now(tz=us_eastern)
    output_dir = f"figures_comparison/{now.strftime('%Y.%m.%d')}/tables"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results storage
    results = {}
    
    for env_name in envs:
        print(f"\nProcessing {env_name}...")
        results[env_name] = {}
        
        for metric_name in metrics:
            print(f"    Processing {metric_name}...")
            
            # Load data from .npz files
            ewa_data, odt_data = load_npz_data(env_name, metric_name)
            
            # Extract metric values
            ewa_final, odt_final = get_metric_values(ewa_data, odt_data)
            
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
    txt_file = f"{output_dir}/comparison_table_from_data_{timestamp.strftime('%H%M%S')}.txt"
    
    with open(txt_file, 'w') as f:
        f.write("EWA-ODT vs ODT Comparison Table (from .npz data)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} ET\n")
        f.write(f"Data source: {data_comparison_base}\n")
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
    csv_file = f"{output_dir}/comparison_table_from_data_{timestamp.strftime('%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\nComparison table saved to:")
    print(f"  Text: {txt_file}")
    print(f"  CSV: {csv_file}")

def main():
    """Main function to generate comparison table from .npz data."""
    print("Generating EWA-ODT vs ODT comparison table from .npz files...")
    print("Metrics: evaluation and augmentation (excluding EWA-specific)")
    print(f"Data source: {data_comparison_base}")
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
