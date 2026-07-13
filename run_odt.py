
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

exp_name = "odt" ## !!! manually specify odt or ewa !!!

# Define RTG values and environment-specific parameters for each environment
ENV_CONFIG = {
    "halfcheetah-medium-expert-v2": {
        "online_rtg": 12000,
        "eval_rtg": 6000,
        "eval_context_length": 5,
        "ordering": 0  # no positional embedding
    },
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

# # Add experiment for max_iters=30 
# for seed in range(1, 11):  # 10 seeds
#     experiments.append({
#         "exp_name": exp_name,
#         "name_run": "normalized_return_vs_steps_30iter",
#         "max_online_iters": 30,
#         "num_updates_per_online_iter": 100,
#         "eval_interval": 2,
#         "envs": ["halfcheetah-medium-expert-v2", "walker2d-medium-replay-v2", "hopper-medium-v2"],
#         "seed": seed
#     })

# Run with max_iters=100 for different seeds
# Results for intermediate iterations will be captured in TensorBoard logs
for seed in range(1, 11):  # 10 seeds
    experiments.append({
        "exp_name": exp_name,
        "name_run": "normalized_return_vs_steps",
        "max_online_iters": 100,  # Will capture results for all intermediate iterations
        "num_updates_per_online_iter": 100,
        "eval_interval": 2,  # Evaluate every 2 iterations to get good granularity
        "envs": ["halfcheetah-medium-expert-v2", "walker2d-medium-replay-v2", "hopper-medium-v2"],
        "seed": seed
    })

# Vary number of online rollouts
for rollouts in range(1, 11, 1):
    for seed in range(1, 11):
        experiments.append({
            "exp_name": exp_name,
            "name_run": "return_vs_online_rollouts",
            "max_online_iters": 30,
            "num_online_rollouts": rollouts,
            "eval_interval": 2,
            "envs": ["halfcheetah-medium-expert-v2", "walker2d-medium-replay-v2", "hopper-medium-v2"],
            "seed": seed  
        })

# Vary training iterations per online iter
for updates in range(50, 501, 50):
    for seed in range(1, 11):
        experiments.append({
            "exp_name": exp_name,
            "name_run": "action_entropy_vs_training",
            "max_online_iters": 30,
            "num_updates_per_online_iter": updates,
            "eval_interval": 3,
            "envs": ["halfcheetah-medium-expert-v2", "walker2d-medium-replay-v2", "hopper-medium-v2"],
            "seed": seed  
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
        )
        
        if "num_updates_per_online_iter" in run_config:
            command += f"--num_updates_per_online_iter {run_config['num_updates_per_online_iter']} "
        if "num_online_rollouts" in run_config:
            command += f"--num_online_rollouts {run_config['num_online_rollouts']} "
        
        # Run experiment and log results
        print(f"Running: {command}")
        
        # Get the path of the folder created by logger.py
        now = datetime.now(tz=us_eastern).strftime("%Y.%m.%d/%H%M")
        exp_folder = f"{results_dir}/{now}_{run_config['exp_name']}_{env}_seed_{run_config['seed']}"
        info_folder = os.path.join(exp_folder, "info")
        os.makedirs(info_folder, exist_ok=True)
        
        # Save config.json in info folder 
        config_path = os.path.join(info_folder, "config.json")
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=4)
        
        # Run command and save output to log.txt
        # with open(os.path.join(info_folder, "log.txt"), "w") as log_file:
        #     subprocess.run(command, shell=True, stdout=log_file, stderr=log_file)

        # Run command and save output to log.txt while also showing in terminal
        log_path = os.path.join(info_folder, "log.txt")
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read and log output in real-time
            for line in process.stdout:
                print(line, end='')  # Print to terminal
                log_file.write(line)  # Write to file
                log_file.flush()  # Ensure immediate writing
            
            # Wait for process to complete
            process.wait()
            
            # Check if process failed
            if process.returncode != 0:
                print(f"Process failed with return code {process.returncode}")
                sys.exit(process.returncode)

# Run all experiments
for exp in experiments:
    run_experiment(exp)

print("All experiments completed.")
