# run multiple experiments with different hyperparameters
# using pred_from_archs_with_zero_cost.py runned with different arguments

import itertools
import subprocess
import os
import sys

# hyperparameter grids

train_sizes = [1_000, 10_000]
connected_feature = [True, False]
arch_index_feature = [True, False]
edge_features_onehot = [True, False]
zero_cost_features = [['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'], []]

script = "pred_from_archs_with_zero_cost.py"

# path to the current Python (from venv if activated)
python_exec = sys.executable

# iterate over all combinations
for ts, cf, aif, efo, zcf in itertools.product(train_sizes, connected_feature, arch_index_feature, edge_features_onehot, zero_cost_features):
    cmd = [python_exec, script, f"--train_size={ts}"]
    if cf:
        cmd.append("--connected_feature")
    if aif:
        cmd.append("--arch_index_feature")
    if efo:
        cmd.append("--edge_features_onehot")
    if zcf:
        cmd.append(f"--zero_cost_features")
        cmd.extend(zcf)
        
    zcf_bool = "withZC" if zcf else "noZC"

    cmd.append(f"--logdir=logs/train{ts}_cf{cf}_aif{aif}_efo{efo}_{zcf_bool}")
    cmd.append("--verbose")

    print("Running:", " ".join(cmd))

    logname = f"logs/train{ts}_cf{cf}_aif{aif}_efo{efo}_{zcf_bool}/logs.txt"
    os.makedirs(os.path.dirname(logname), exist_ok=True)
    with open(logname, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


