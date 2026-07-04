import os
import sys
import subprocess

def run_experiment(logdir: str, script: str="tabpfn_with_nats_bench/pred_from_archs_with_zero_cost_and_train_acc.py", **kwargs):
    """
    Run a Python script with specified arguments and log the output.

    Args:
        logdir: Directory where logs will be saved.
        script: Path to the Python script to run, default is "tabpfn_with_nats_bench/pred_from_archs_with_zero_cost_and_train_acc.py".
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

    cmd.append(f"--logdir={logdir}")
    cmd.append("--verbose")

    # Log file path
    logname = os.path.join(logdir, "logs.txt")

    # Run the command and log output
    with open(logname, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        
