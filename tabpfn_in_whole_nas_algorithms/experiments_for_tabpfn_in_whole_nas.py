import subprocess
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Run an experiment with the evolutionary or active learning algorithm')
parser.add_argument('--logdir', type=str, default='tabpfn_in_whole_nas_algorithms/logs/', help='Directory to save logs.')
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 200, 300, 400, 1500, 1600, 1700, 2500, 2600, 2700], help='List of random seeds to run the experiment with.')
parser.add_argument('--script_dir', type=str, default='tabpfn_in_whole_nas_algorithms/', help='Path to the Python script to run.')
parser.add_argument('--search_algorithm', type=str, default='active_learning', choices=['evolution', 'active_learning'], help='Search algorithm to use: evolution or active_learning')

def run_experiment(logdir: str, script: str="tabpfn_in_whole_nas_algorithms/evolution.py", inform_about_progress: bool=True, **kwargs):
    """
    Run a Python script with specified arguments and log the output.

    Args:
        logdir: Directory where logs will be saved.
        script: Path to the Python script to run, default is "tabpfn_in_whole_nas_algorithms/evolution.py".
        inform_about_progress: Whether to print progress information.
        kwargs: Additional command-line arguments to pass to the script.
                Boolean arguments should be passed as True/False flags,
                list arguments should be passed as lists
                and other arguments should be passed as key-value pairs.
    """
    # Ensure the log directory exists
    os.makedirs(logdir, exist_ok=True)

    # Build the command
    cmd = [sys.executable, script]
    
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            cmd.append(f"--{key}")
            cmd.extend(value)
        else:
            cmd.append(f"--{key}={value}")

    if inform_about_progress:
        print("Running:", " ".join(cmd))

    # Log file path
    logname = os.path.join(logdir, "logs.txt")

    # Run the command and log output
    with open(logname, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.search_algorithm == "evolution":
        script = os.path.join(args.script_dir, "evolution.py")
        logdir = os.path.join(args.logdir, "evolution")
        for seed in args.seeds:
            seed_logdir = os.path.join(logdir, f"seed_{seed}")
            run_experiment(logdir=seed_logdir, script=script, seed=seed)
    else:
        script = os.path.join(args.script_dir, "active_learning.py")
        logdir = os.path.join(args.logdir, "active_learning")
        for seed in args.seeds:
            seed_logdir = os.path.join(logdir, f"seed_{seed}")
            run_experiment(logdir=seed_logdir, script=script, seed=seed, plot=True, plot_dir=seed_logdir)










