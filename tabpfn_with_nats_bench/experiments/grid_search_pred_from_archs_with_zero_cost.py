# run multiple experiments with different hyperparameters
# using pred_from_archs_with_zero_cost_and_train_acc.py runned with different arguments

import itertools
import subprocess
import os
import sys

# hyperparameter grids

seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900]
train_sizes = [10, 100, 1000, 10_000]
connected_feature = [True, False]
arch_index_feature = [True, False]
edge_features = ["none", "raw", "onehot"]
zero_cost_features = [['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'], []]
skip_unconnected = [True, False]
epochs_n_features = [0, 10]


script = "tabpfn_with_nats_bench/pred_from_archs_with_zero_cost_and_train_acc.py"

# path to the current Python (from venv if activated)
python_exec = sys.executable

# iterate over all combinations
for s, ts, cf, aif, ef, zcf, su, enf in itertools.product(seeds, train_sizes, connected_feature, arch_index_feature, edge_features, zero_cost_features, skip_unconnected, epochs_n_features):
    # skip the combination where no features are included
    if cf == False and aif == False and ef == "none" and zcf == [] and enf == 0:
        continue

    # build the command
    cmd = [python_exec, script, f"--train_size={ts}", f"--seed={s}"]
    if cf:
        cmd.append("--connected_feature")
    if aif:
        cmd.append("--arch_index_feature")
    cmd.append(f"--edge_features={ef}")
    if zcf:
        cmd.append(f"--zero_cost_features")
        cmd.extend(zcf)
    if su:
        cmd.append("--skip_unconnected")
    if enf > 0:
        cmd.append(f"--epochs_n_features={enf}")
        
    zcf_bool = "withZC" if zcf else "noZC"
    su_bool = "skipUnconnected" if su else "noSkipUnconnected"

    cmd.append(f"--logdir=tabpfn_with_nats_bench/logs/grid_search_results/seed_{s}/train-{ts}_cf-{cf}_aif-{aif}_ef-{ef}_{zcf_bool}_{su_bool}_enf-{enf}")
    cmd.append("--verbose")

    print("Running:", " ".join(cmd))

    logname = f"tabpfn_with_nats_bench/logs/grid_search_results/seed_{s}/train-{ts}_cf-{cf}_aif-{aif}_ef-{ef}_{zcf_bool}_{su_bool}_enf-{enf}/logs.txt"
    os.makedirs(os.path.dirname(logname), exist_ok=True)
    with open(logname, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


