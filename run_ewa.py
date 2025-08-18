import os
import subprocess
import json
import time
from datetime import datetime
import pytz
from logger import Logger
import sys
# Set timezone to US Eastern Timezone
us_eastern = pytz.timezone('US/Eastern')

# Define base parameters
base_command = "python main.py"
results_dir = "./exp"
os.makedirs(results_dir, exist_ok=True)

# EWA Configuration
EWA_CONFIG = {
    "beta": 0.05,
    "phi": 0.05,
    "delta": 0.8,
    "trajectory_length": 1000,
    "num_codes": 16,
    "grid_bins_factor": 1.0,  # Adaptive: 1.0 bins per action dimension
}

exp_name = "ewa" ## !!! manually specify odt or ewa !!!

# Define RTG values and environment-specific parameters for each environment
ENV_CONFIG = {
    "walker2d-medium-replay-v2": {
        "online_rtg": 10000,
        "eval_rtg": 5000,
        "eval_context_length": 5,
        "ordering": 0  # no positional embedding
    },
    "hopper-medium-v2": {
        "online_rtg": 7200,
        "eval_rtg": 3600,
        "eval_context_length": 20,
        "ordering": 1  # yes positional embedding
    }
}

experiments = []

# Add experiment for max_iters=20 
for seed in range(1, 5):  # 4 seeds
    experiments.append({
        "exp_name": exp_name,
        "name_run": "normalized_return_vs_steps_20iter",
        "max_online_iters": 20,
        "num_updates_per_online_iter": 100,
        "eval_interval": 2,
        "envs": ["hopper-medium-v2", "walker2d-medium-replay-v2"],  # Removed halfcheetah (6D) 
        "seed": seed,
        **EWA_CONFIG  # Unpack EWA parameters
    })

# Run with max_iters=100 for different seeds
# Results for intermediate iterations will be captured in TensorBoard logs
for seed in range(1, 6):  # 5 seeds
    experiments.append({
        "exp_name": exp_name, 
        "name_run": "normalized_return_vs_steps",
        "max_online_iters": 100,  # Will capture results for all intermediate iterations
        "num_updates_per_online_iter": 100,
        "eval_interval": 2,  # Evaluate every 2 iterations to get good granularity
        "envs": ["hopper-medium-v2", "walker2d-medium-replay-v2"],  # Removed halfcheetah (6D)
        "seed": seed,
        **EWA_CONFIG  # Unpack EWA parameters
    })

# Vary number of online rollouts
for rollouts in range(1, 6, 1):
    for seed in range(1, 6):
        experiments.append({
            "exp_name": exp_name,
            "name_run": "return_vs_online_rollouts",
            "max_online_iters": 30,
            "num_online_rollouts": rollouts,
            "eval_interval": 2,
            "envs": ["hopper-medium-v2", "walker2d-medium-replay-v2"],  # Removed halfcheetah (6D)
            "seed": seed,
            **EWA_CONFIG  # Unpack EWA parameters
        })

# Vary training iterations per online iter
for updates in range(50, 501, 50):
    for seed in range(1, 6):
        experiments.append({
            "exp_name": exp_name,
            "name_run": "action_entropy_vs_training",
            "max_online_iters": 30,
            "num_updates_per_online_iter": updates,
            "eval_interval": 3,
            "envs": ["hopper-medium-v2", "walker2d-medium-replay-v2"],  # Removed halfcheetah (6D)
            "seed": seed,
            **EWA_CONFIG  # Unpack EWA parameters
        })

# Function to run experiment
def run_experiment(run_config):
    for env in run_config["envs"]:
        # Get environment-specific values
        env_config = ENV_CONFIG[env]
        
        # Construct command
        command = [
            "python", "main.py",
            "--env", env,
            "--max_online_iters", str(run_config['max_online_iters']),
            "--eval_interval", str(run_config['eval_interval']),
            "--seed", str(run_config['seed']),
            "--exp_name", run_config['exp_name'],
            "--online_rtg", str(env_config['online_rtg']),
            "--eval_rtg", str(env_config['eval_rtg']),
            "--eval_context_length", str(env_config['eval_context_length']),
            "--ordering", str(env_config['ordering']),
            "--beta", str(run_config['beta']),
            "--phi", str(run_config['phi']),
            "--delta", str(run_config['delta']),
            "--trajectory_length", str(run_config['trajectory_length']),
            "--num_codes", str(run_config['num_codes']),
            "--grid_bins_factor", str(run_config['grid_bins_factor']),
        ]
        
        if "num_updates_per_online_iter" in run_config:
            command.extend(["--num_updates_per_online_iter", str(run_config['num_updates_per_online_iter'])])
        if "num_online_rollouts" in run_config:
            command.extend(["--num_online_rollouts", str(run_config['num_online_rollouts'])])
        
        # Run experiment and log results
        print(f"Running: {' '.join(command)}")
        print(f"Starting experiment at {datetime.now(tz=us_eastern).strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        # Get the path of the folder created by logger.py
        now = datetime.now(tz=us_eastern).strftime("%Y.%m.%d/%H%M")
        exp_folder = f"{results_dir}/{now}_{run_config['exp_name']}_{env}_seed_{run_config['seed']}"
        info_folder = os.path.join(exp_folder, "info")
        os.makedirs(info_folder, exist_ok=True)
        
        # Save config.json in info folder 
        config_path = os.path.join(info_folder, "config.json")
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=4)
        
        # Run command and save output to log.txt while also showing in terminal
        log_path = os.path.join(info_folder, "log.txt")
        
        # Use the clean approach from run_quick1_ewa.py
        env_vars = os.environ.copy()
        env_vars["PYTHONUNBUFFERED"] = "1"
        
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env_vars
            )
            
            last = time.time()
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
                if time.time() - last > 300:  # 5 minutes
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Process still running...")
                    last = time.time()
            
            process.wait()
            return_code = process.returncode
        
        # Check if process failed
        if return_code != 0:
            print(f"Process failed with return code {return_code}")
            sys.exit(return_code)

# Run all experiments
for exp in experiments:
    run_experiment(exp)

print("All experiments completed.")
