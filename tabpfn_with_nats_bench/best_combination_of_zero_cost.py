# run multiple experiments with different hyperparameters
# using pred_from_archs_with_zero_cost.py runned with different arguments

import itertools
import subprocess
import os
import sys

# hyperparameter grids

train_size = 1000
connected_feature = True
zero_cost_features = ['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn']

script = "pred_from_archs_with_zero_cost.py"

# path to the current Python (from venv if activated)
python_exec = sys.executable

# generate all combinations of zero_cost_features
feature_sets = []
for r in range(len(zero_cost_features) + 1):
    for combo in itertools.combinations(zero_cost_features, r):
        feature_sets.append(list(combo))

# iterate over all combinations
for zcf in feature_sets:
    cmd = [python_exec, script, f"--train_size={train_size}"]
    cmd.append("--connected_feature")
    cmd.append(f"--zero_cost_features")
    cmd.extend(zcf)

    cmd.append(f"--logdir=logs/zero_cost_combination/{'_'.join(zcf) if zcf else 'noZC'}")
    cmd.append("--verbose")

    print("Running:", " ".join(cmd))

    logname = f"logs/zero_cost_combination/{'_'.join(zcf) if zcf else 'noZC'}/logs.txt"
    os.makedirs(os.path.dirname(logname), exist_ok=True)
    with open(logname, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)

