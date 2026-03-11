import numpy as np
import os
import tabpfn_with_nats_bench.utils.get_important_lines as gil

root_dir = "tabpfn_with_nats_bench/logs/epochs_n_features/"

epochs_n_features = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40]

paths_to_experiments = os.listdir(root_dir)
paths_to_experiments = [p for p in paths_to_experiments if p.endswith("_su_zcf")] # remove dirs that does not end with _su_zcf

mse_table = np.zeros([len(paths_to_experiments), len(epochs_n_features)])
spearman_table = np.zeros([len(paths_to_experiments), len(epochs_n_features)])
kendall_table = np.zeros([len(paths_to_experiments), len(epochs_n_features)])

for run_idx, exp_dir in enumerate(paths_to_experiments):
    path_to_exp_dir = os.path.join(root_dir, exp_dir)
    for subdir in os.listdir(path_to_exp_dir):
        folder_path = os.path.join(path_to_exp_dir, subdir)
        log_path = os.path.join(folder_path, "logs.txt")

        if not os.path.isdir(folder_path) or not os.path.isfile(log_path):
            raise ValueError(f"Expected directory with logs.txt: {folder_path}")

        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()

        # find used epochs_n_features
        enf_value = gil.find_epochs_n_features(log_content)

        # find mean squared error
        mse_value = gil.find_mse(log_content)
        mse_table[run_idx, epochs_n_features.index(enf_value)] = mse_value

        # find spearman's rank correlation coefficient
        spearman_value = gil.find_spearman(log_content)
        spearman_table[run_idx, epochs_n_features.index(enf_value)] = spearman_value

        # find kendall's tau
        kendall_value = gil.find_kendall(log_content)
        kendall_table[run_idx, epochs_n_features.index(enf_value)] = kendall_value

def sort_and_print_results(metric_table, metric_name, sort_ascending=True):
    sorted_results = []
    for idx in range(metric_table.shape[1]):
        metric_values = metric_table[:, idx]
        name = f"{epochs_n_features[idx]} Epochs as Features"
        metric_mean_value = np.mean(metric_values)
        metric_variance_value = np.var(metric_values)
        sorted_results.append((name, metric_mean_value, metric_variance_value))
    sorted_results.sort(key=lambda x: x[1], reverse=not sort_ascending)

    print(f"\n=== {metric_name} Comparison ===")
    for name, mean_value, var_value in sorted_results:
        print(f"{name}: {mean_value:.6f} (Variance: {var_value:.6f})")

# sort by MSE
sort_and_print_results(mse_table, "Mean Squared Error")

# sort by Spearman's Rank Correlation Coefficient
sort_and_print_results(spearman_table, "Spearman's Rank Correlation Coefficient", sort_ascending=False)

# sort by Kendall's Tau
sort_and_print_results(kendall_table, "Kendall's Tau (rank correlation coefficient)", sort_ascending=False)
