import numpy as np
import matplotlib.pyplot as plt
import json
import os
from logger import Logger
from pathlib import Path
from datetime import datetime
import pytz
# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')
now = datetime.now(tz=us_eastern).strftime("%Y.%m.%d/%H")


# Paths to logs and debug data
odt_log_path = "./exp/2025.02.18/13-odt/debug_attention_values.npy/debug_attention_values.npy"
ewa_odt_log_path = "./exp/2025.02.19/22-ewa/debug_attention_values.npy/debug_attention_values.npy"

# Load debug attention values if available
def load_debug_data(file_path):
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True).item()
    else:
        return None

# Load both models' data
odt_debug_data = load_debug_data(odt_log_path)
ewa_odt_debug_data = load_debug_data(ewa_odt_log_path)


# Analyze Attention Changes if debug data exists
if odt_debug_data and ewa_odt_debug_data:
    attention_before_odt = np.mean(odt_debug_data["attention_scores"])
    attention_before_ewa = np.mean(ewa_odt_debug_data["attention_scores_before"])
    attention_after_ewa = np.mean(ewa_odt_debug_data["attention_scores_after"])
    print(f"ODT: attention_before_odt {attention_before_odt}")
    print(f"EWA-ODT: attention_before_odt {attention_before_ewa}")
    print(f"EWA-ODT: attention_after_ewa {attention_after_ewa}")

    # Plot Attention Score Changes
    plt.figure(figsize=(8, 5))
    plt.bar(["ODT Before", "EWA Before", "EWA After"], [attention_before_odt, attention_before_ewa, attention_after_ewa], color=["blue", "green", "red"])
    plt.ylabel("Mean Attention Score")
    plt.title("Attention Score Comparison")
    plt.grid(axis="y")
    # plt.show()
    now = datetime.now(tz=us_eastern).strftime("%Y.%m.%d_%H%M")
    plt.savefig(f"./exp/figures/{now}.png")
else:
    print("Debug data for attention scores not found. Skipping attention analysis.")
