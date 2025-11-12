import numpy as np
import os
import regex as re

root_dir = "logs/zero_cost_combination"

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

for subdir in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, subdir)
    log_path = os.path.join(folder_path, "logs.txt")

    if not os.path.isdir(folder_path) or not os.path.isfile(log_path):
        raise ValueError(f"Expected directory with logs.txt: {folder_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # find used zero-cost features
    zcf_match = re.search(r"included zero-cost features:\n((?:\s+- .+\n)+)", log_content)
    if zcf_match:
        zcf_lines = zcf_match.group(1).strip().split("\n")
        zcf_list = [line.split(":")[0].strip().lstrip("- ").strip() for line in zcf_lines]
        mse_table_idx = tuple(1 if feat in zcf_list else 0 for feat in zero_cost_features)
    else:
        mse_table_idx = tuple(0 for _ in zero_cost_features)

    # find line containing "Mean Squared Error:"
    mse_match = re.search(r"Mean Squared Error:\s*([0-9.]+)", log_content)
    if mse_match:
        mse_value = float(mse_match.group(1))
        mse_table[mse_table_idx] = mse_value
    

# sort by MSE
sorted_results = []
for idx in np.ndindex(mse_table.shape):
    zcf_included = [zero_cost_features[i] for i in range(len(zero_cost_features)) if idx[i] == 1]
    name = ", ".join([zero_cost_features_names[feat] for feat in zcf_included]) if zcf_included else "No Zero-Cost Features"
    mse_value = mse_table[idx]
    sorted_results.append((name, mse_value))
sorted_results.sort(key=lambda x: x[1])

print("\n=== MSE Comparison ===")
for name, mse in sorted_results:
    print(f"{name}: {mse:.6f}")

best = sorted_results[0]
print(f"\nLowest MSE: {best[0]} ({best[1]:.6f})")
