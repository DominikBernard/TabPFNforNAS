import numpy as np
import os
import regex as re
import get_important_lines as gil

root_dir = "logs/zero_cost_combination/seed_42_su"

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

mse_table = np.zeros([2 for _ in range(len(zero_cost_features))])
spearman_table = np.zeros([2 for _ in range(len(zero_cost_features))])
kendall_table = np.zeros([2 for _ in range(len(zero_cost_features))])

for subdir in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, subdir)
    log_path = os.path.join(folder_path, "logs.txt")

    if not os.path.isdir(folder_path) or not os.path.isfile(log_path):
        raise ValueError(f"Expected directory with logs.txt: {folder_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # find used zero-cost features
    zcf_list = gil.find_zero_cost_features(log_content)
    mse_table_idx = tuple(1 if feat in zcf_list else 0 for feat in zero_cost_features)

    # find mean squared error
    mse_value = gil.find_mse(log_content)
    mse_table[mse_table_idx] = mse_value

    # find spearman's rank correlation coefficient
    spearman_value = gil.find_spearman(log_content)
    spearman_table[mse_table_idx] = spearman_value

    # find kendall's tau
    kendall_value = gil.find_kendall(log_content)
    kendall_table[mse_table_idx] = kendall_value
    
def sort_and_print_results(metric_table, metric_name, sort_ascending=True):
    sorted_results = []
    for idx in np.ndindex(metric_table.shape):
        zcf_included = [zero_cost_features[i] for i in range(len(zero_cost_features)) if idx[i] == 1]
        name = ", ".join([zero_cost_features_names[feat] for feat in zcf_included]) if zcf_included else "No Zero-Cost Features"
        metric_value = metric_table[idx]
        sorted_results.append((name, metric_value))
    sorted_results.sort(key=lambda x: x[1], reverse=not sort_ascending)

    print(f"\n=== {metric_name} Comparison ===")
    for name, value in sorted_results:
        print(f"{name}: {value:.6f}")

    best = sorted_results[0]
    print(f"\nBest {metric_name}: {best[0]} ({best[1]:.6f})")
    print()

# sort by MSE
sort_and_print_results(mse_table, "Mean Squared Error")

# sort by Spearman's Rank Correlation Coefficient
sort_and_print_results(spearman_table, "Spearman's Rank Correlation Coefficient", sort_ascending=False)

# sort by Kendall's Tau
sort_and_print_results(kendall_table, "Kendall's Tau (rank correlation coefficient)", sort_ascending=False)
