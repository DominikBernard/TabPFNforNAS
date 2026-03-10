import numpy as np
import os
from collections import defaultdict

import utils.get_important_lines as gil


root_dir = "tabpfn_with_nats_bench/logs/zero_cost_combination/"

zero_cost_features = ['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn']
zero_cost_features_names = {
    'f': 'Fisher',
    'g_n': 'GradNorm',
    'g': 'Grasp',
    'j_c': 'JacobCov',
    'p': 'Plain',
    's': 'SNIP',
    'sy': 'Synflow',
    'sy_bn': 'SynflowBN'
}

paths_to_experiments = os.listdir(root_dir)
paths_to_experiments = [p for p in paths_to_experiments if p.endswith("_su")] # remove dirs that does not end with _su

mse_table = defaultdict(list)
spearman_table = defaultdict(list)
kendall_table = defaultdict(list)

for run_idx, exp_dir in enumerate(paths_to_experiments):
    path_to_exp_dir = os.path.join(root_dir, exp_dir)
    for subdir in os.listdir(path_to_exp_dir):
        folder_path = os.path.join(path_to_exp_dir, subdir)
        log_path = os.path.join(folder_path, "logs.txt")

        if not os.path.isdir(folder_path) or not os.path.isfile(log_path):
            raise ValueError(f"Expected directory with logs.txt: {folder_path}")

        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()

        # find used zero-cost features
        zcf_list = gil.find_zero_cost_features(log_content)
        
        # find mean squared error
        mse_value = gil.find_mse(log_content)
        mse_table[tuple(sorted(zcf_list))].append(mse_value)

        # find spearman's rank correlation coefficient
        spearman_value = gil.find_spearman(log_content)
        spearman_table[tuple(sorted(zcf_list))].append(spearman_value)

        # find kendall's tau
        kendall_value = gil.find_kendall(log_content)
        kendall_table[tuple(sorted(zcf_list))].append(kendall_value)

def sort_and_print_results(metric_table, metric_name, sort_ascending=True):
    sorted_results = []
    for zcf_tuple, metric_values in metric_table.items():
        name = ", ".join([zero_cost_features_names[feat] for feat in zcf_tuple]) if zcf_tuple else "No Zero-Cost Features"
        metric_mean_value = np.mean(metric_values)
        metric_variance_value = np.var(metric_values)
        sorted_results.append((name, metric_mean_value, metric_variance_value))
    sorted_results.sort(key=lambda x: x[1], reverse=not sort_ascending)

    print(f"\n=== {metric_name} Comparison ===")
    for name, value, var in sorted_results:
        print(f"{name}: {value:.6f} (Variance: {var:.6f})")

    best = sorted_results[0]
    print(f"\nBest {metric_name}: {best[0]} ({best[1]:.6f})")
    

# sort by MSE
sort_and_print_results(mse_table, "Mean Squared Error")

# sort by Spearman's Rank Correlation Coefficient
sort_and_print_results(spearman_table, "Spearman's Rank Correlation Coefficient", sort_ascending=False)

# sort by Kendall's Tau
sort_and_print_results(kendall_table, "Kendall's Tau (rank correlation coefficient)", sort_ascending=False)

