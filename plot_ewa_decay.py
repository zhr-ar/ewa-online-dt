import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import os

# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')

# EWA parameters from ewa_decay.py
phi = 0.9
initial_delta = 0.7
final_delta = 0.1
max_online_iters = 100
num_updates_per_online_iter = 100
batch_size = 256
replay_size = 1000
trajectory_length = 1000

# Calculate number of revisits and decay step
n_visits = (max_online_iters * num_updates_per_online_iter * batch_size) // (replay_size * trajectory_length)
decay_step = max_online_iters / n_visits if n_visits > 0 else max_online_iters

# Generate steps array
steps = np.arange(max_online_iters)

# Calculate decay ratio and delta values
decay_ratios = np.minimum(1.0, steps / decay_step)
delta_values = initial_delta * (1 - decay_ratios) + final_delta * decay_ratios

# Create the plot
plt.figure(figsize=(12, 7))

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

# Add title with parameters
title = f'Theoretical Delta and Decay Ratio Over Training\n'
title += f'Parameters: phi={phi}, initial_delta={initial_delta}, final_delta={final_delta}\n'
title += f'n_visits={n_visits:.1f}, decay_step={decay_step:.1f}'
plt.title(title)

# Add grid
plt.grid(True)

# Create directory structure with date and time
now = datetime.now(tz=us_eastern)
exp_folder = f"figures_comparison/{now.strftime('%Y.%m.%d')}/delta"
os.makedirs(exp_folder, exist_ok=True)

# Save the figure
plt.savefig(f"{exp_folder}/{now.strftime('%H%M%S')}_theoretical_delta_and_decay_ratio.png")
plt.close()

# Print parameters and calculations
print("\nEWA Decay Parameters and Calculations:")
print("=" * 50)
print(f"Number of revisits (n_visits): {n_visits:.1f}")
print(f"Decay step: {decay_step:.1f}")
print(f"Initial delta: {initial_delta}")
print(f"Final delta: {final_delta}")
print(f"Phi: {phi}")
print(f"Max online iterations: {max_online_iters}")
print(f"Updates per iteration: {num_updates_per_online_iter}")
print(f"Batch size: {batch_size}")
print(f"Replay size: {replay_size}")
print(f"Trajectory length: {trajectory_length}")
print("=" * 50) 