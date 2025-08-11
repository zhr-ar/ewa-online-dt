import os, subprocess, time

env_name = "walker2d-medium-replay-v2"
seeds = [1, 2, 3]

# Fast settings for a quick but meaningful signal
common_args = [
    "--max_online_iters", "10",
    "--eval_interval", "2",
    "--num_updates_per_online_iter", "30",
    "--batch_size", "64",
    "--K", "20",
    "--eval_context_length", "5",
    "--ordering", "0",                 # walker2d without positional embedding
    "--online_rtg", "10000",
    "--eval_rtg", "5000",
    "--exp_name", "ewa",
    # EWA params
    "--beta", "0.05",
    "--phi", "0.05",
    "--delta", "0.8",
    "--trajectory_length", "1000",
    "--num_codes", "27",
    "--grid_bins_factor", "1.0",
    "--device", "cuda",
]

def run_seed(seed: int):
    cmd = ["python", "-u", "main.py", "--env", env_name, "--seed", str(seed)] + common_args
    print("Running:", " ".join(cmd))
    log_dir = "./exp/quick_runs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{env_name}_ewa_seed_{seed}.log")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        last = time.time()
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
            if time.time() - last > 300:
                print(f"[{time.strftime('%H:%M:%S')}] Process still running...")
                last = time.time()
        proc.wait()
    print(f"Finished seed {seed} with return code {proc.returncode}. Log: {log_path}")

if __name__ == "__main__":
    for s in seeds:
        run_seed(s)