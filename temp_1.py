import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_event_logs(folder_path):
    """Load EWA metrics from event files."""
    ea = event_accumulator.EventAccumulator(
        folder_path,
        size_guidance={
            event_accumulator.SCALARS: 0,
        }
    )
    ea.Reload()
    
    # Get all scalar tags
    tags = ea.Tags()['scalars']
    
    # Filter for EWA metrics
    ewa_metrics = {
        'delta': [],
        'decay_ratio': [],
        'attraction': []
    }
    
    # Load data for each layer
    for tag in tags:
        if 'delta_layer_' in tag:
            layer_idx = int(tag.split('_')[-1])
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ewa_metrics['delta'].append((layer_idx, steps, values))
        elif 'decay_ratio_layer_' in tag:
            layer_idx = int(tag.split('_')[-1])
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ewa_metrics['decay_ratio'].append((layer_idx, steps, values))
        elif 'attraction_layer_' in tag:
            layer_idx = int(tag.split('_')[-1])
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ewa_metrics['attraction'].append((layer_idx, steps, values))
    
    return ewa_metrics

def plot_ewa_metrics(ewa_metrics, save_path=None):
    """Plot EWA metrics for each layer."""
    metrics = ['delta', 'decay_ratio', 'attraction']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    colors = ['blue', 'red', 'green', 'purple']  # Colors for different layers
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for layer_idx, steps, values in ewa_metrics[metric]:
            ax.plot(steps, values, label=f'Layer {layer_idx}', color=colors[layer_idx])
        
        ax.set_title(f'{metric.capitalize()} Values Over Time')
        ax.set_xlabel('Steps')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    import tensorflow as tf
    import os

    event_file = '/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.05.08/2046_ewa_hopper-medium-v2_seed_1/events.out.tfevents.1746751922.ip-172-31-46-240.14899.1'

    print("Reading event file:", event_file)
    print("\nAvailable metrics:")
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            print(f"Tag: {value.tag}, Value: {value.simple_value if hasattr(value, 'simple_value') else 'No simple value'}") 

            
    # Path to your event files
    event_path = '/home/ubuntu/online_decision_transformer/ewa-online-dt-main/exp/2025.05.08/2046_ewa_hopper-medium-v2_seed_1'
    
    # Load the metrics
    ewa_metrics = load_event_logs(event_path)
    
    # Plot the metrics
    plot_ewa_metrics(ewa_metrics, save_path='ewa_metrics_plot.png')

if __name__ == "__main__":
    main() 