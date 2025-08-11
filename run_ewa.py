import os
import subprocess
import json
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
for seed in range(2, 6):  # 5 seeds
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
        command = (
            f"{base_command} --env {env} "
            f"--max_online_iters {run_config['max_online_iters']} "
            f"--eval_interval {run_config['eval_interval']} "
            f"--seed {run_config['seed']} "
            f"--exp_name {run_config['exp_name']} "
            f"--online_rtg {env_config['online_rtg']} "
            f"--eval_rtg {env_config['eval_rtg']} "
            f"--eval_context_length {env_config['eval_context_length']} "
            f"--ordering {env_config['ordering']} "
            f"--beta {run_config['beta']} "
            f"--phi {run_config['phi']} "
            f"--delta {run_config['delta']} "
            f"--trajectory_length {run_config['trajectory_length']} "
            f"--num_codes {run_config['num_codes']} "
            f"--grid_bins_factor {run_config['grid_bins_factor']} "
        )
        
        if "num_updates_per_online_iter" in run_config:
            command += f"--num_updates_per_online_iter {run_config['num_updates_per_online_iter']} "
        if "num_online_rollouts" in run_config:
            command += f"--num_online_rollouts {run_config['num_online_rollouts']} "
        
        # Run experiment and log results
        print(f"Running: {command}")
        print(f"Starting experiment at {datetime.now(tz=us_eastern).strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"Log file: {log_path}")
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
        
        # Improved approach: run with real-time output and proper buffering
        def run_with_output(command, log_file):
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,  # No buffering
                universal_newlines=True,
                env=env
            )
            
            # Set up non-blocking output reading
            import select
            import sys
            
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    # Read any remaining output
                    remaining_output = process.stdout.read()
                    if remaining_output:
                        print(remaining_output, end='')
                        log_file.write(remaining_output)
                        log_file.flush()
                    break
                
                # Use select to check if there's data to read (non-blocking)
                if select.select([process.stdout], [], [], 0.1)[0]:
                    output = process.stdout.readline()
                    if output:
                        print(output.rstrip())  # Print to terminal immediately
                        log_file.write(output)  # Write to file
                        log_file.flush()  # Ensure immediate writing
                else:
                    # No output available, but process is still running
                    # Print a progress indicator every 60 seconds
                    import time
                    if not hasattr(run_with_output, 'last_progress'):
                        run_with_output.last_progress = time.time()
                    
                    if time.time() - run_with_output.last_progress > 360:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Process still running...")
                        run_with_output.last_progress = time.time()
            
            return process.poll()
        
        with open(log_path, "w") as log_file:
            return_code = run_with_output(command, log_file)
        
        # Check if process failed
        if return_code != 0:
            print(f"Process failed with return code {return_code}")
            sys.exit(return_code)

# Run all experiments
for exp in experiments:
    run_experiment(exp)

print("All experiments completed.")
