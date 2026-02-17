import numpy as np
import os
import regex as re

root_dir = "logs/epochs_n_features/"

epochs_n_features = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40]

paths_to_experiments = os.listdir(root_dir)
paths_to_experiments = [p for p in paths_to_experiments if p.endswith("_su_zcf")] # remove dirs that does not end with _su_zcf

mse_table = np.zeros([len(paths_to_experiments), len(epochs_n_features)])

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
        enf_match = m = re.search(r'epochs_n_features:\s*(\d+)', log_content)
        
        # find line containing "Mean Squared Error:"
        mse_match = re.search(r"Mean Squared Error:\s*([0-9.]+)", log_content)

        if enf_match and mse_match:
            mse_value = float(mse_match.group(1))
            enf_value = int(enf_match.group(1))
            if enf_value in epochs_n_features:
                mse_table[run_idx, epochs_n_features.index(enf_value)] = mse_value
            else:
                raise ValueError(f"Unexpected epochs_n_features value: {enf_value} in log: {log_path}")
        else:
            raise ValueError(f"Could not find epochs_n_features or MSE in log: {log_path}")
    

# sort by MSE
sorted_results = []
for idx in range(mse_table.shape[1]):
    mse_values = mse_table[:, idx]
    name = f"{epochs_n_features[idx]} Epochs as Features"
    mse_mean_value = np.mean(mse_values)
    mse_variance_value = np.var(mse_values)
    sorted_results.append((name, mse_mean_value, mse_variance_value))
sorted_results.sort(key=lambda x: x[1])

print("\n=== MSE Comparison ===")
for name, mse, var in sorted_results:
    print(f"{name}: {mse:.6f} (Variance: {var:.6f})")

best = sorted_results[0]
print(f"\nLowest MSE: {best[0]} ({best[1]:.6f})")
