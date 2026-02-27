import itertools
import subprocess
import os
import sys
from typing import Literal

# hyperparameter grids

train_size = 1000
seeds = [42, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
skip_unconnected: Literal["_su", ""] = "" # whether to skip unconnected features (as in the best combination found in the previous experiment)
epochs_n_features = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40]

script = "pred_from_archs_with_zero_cost_and_train_acc.py"

# path to the current Python (from venv if activated)
python_exec = sys.executable

# generate all combinations of epochs_n_features

for seed in seeds:
    # iterate over all combinations
    for enf in epochs_n_features:
        cmd = [python_exec, script, f"--train_size={train_size}"]
        cmd.append(f"--seed={seed}")
        cmd.append(f"--epochs_n_features={enf}")
        if skip_unconnected == "_su":
            cmd.append(f"--skip_unconnected")
        cmd.extend(["--zero_cost_features", 'f', 'g_n', 'g', 'j_c', 'p', 's', 'sy']) # use all zero-cost features except SynflowBN as in the best combination found in the previous experiment

        cmd.append(f"--logdir=logs/epochs_n_features/seed_{seed}{skip_unconnected}_zcf/{enf}_epochs")
        cmd.append("--verbose")

        print("Running:", " ".join(cmd))

        logname = f"logs/epochs_n_features/seed_{seed}{skip_unconnected}_zcf/{enf}_epochs/logs.txt"
        os.makedirs(os.path.dirname(logname), exist_ok=True)
        with open(logname, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)

