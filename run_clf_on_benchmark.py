import openml
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier


# Load the benchmark suite
# id 293 corresponds to the OpenML benchmark suite AutoML Benchmark Training Datasets
suite_id = 293  # id of the benchmark suite
benchmark_suite = openml.study.get_suite(suite_id)
random_state = 42
test_size = 0.25

# Initialize a list to store performance metrics
results = []
failures = []

# Retrieve the tasks from the benchmark suite
tasks = benchmark_suite.tasks

# Check if tasks are available
if tasks is None or len(tasks) == 0:
    raise ValueError("No tasks found in the benchmark suite.")

for task_id in tasks:
    try:
        # Get task and dataset
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train model
        clf = TabPFNClassifier()
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

        results.append({
            "task_id": task_id,
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "n_samples": len(X),
            "n_features": X.shape[1] if X.shape else 0,
            "n_classes": len(np.unique(y)),
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        })
        
        print(f"Task {task_id} ({dataset.name}): Accuracy={accuracy:.4f}, ROC AUC={roc_auc:.4f}")
        
    except Exception as e:
        print(f"An error occurred with task {task_id}: {e}")
        failures.append({"task_id": task_id, "error": str(e)})


# --- Convert to DataFrame ---
results_df = pd.DataFrame(results)

# --- Compute Summary Statistics ---
summary = {
    "mean_accuracy": results_df["accuracy"].mean(),
    "std_accuracy": results_df["accuracy"].std(),
    "mean_roc_auc": results_df["roc_auc"].mean(),
    "std_roc_auc": results_df["roc_auc"].std(),
}

# --- Save Results ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_prefix = f"benchmark_{suite_id}_{timestamp}"

# Save CSV
results_df.to_csv(f"results/{filename_prefix}_results.csv", index=False)

# Save metadata
metadata = {
    "benchmark_suite_id": suite_id,
    "benchmark_suite_name": benchmark_suite.name,
    "run_timestamp": timestamp,
    "algorithm": "TabPFNClassifier",
    "algorithm_params": "default",
    "sklearn_random_state": random_state,
    "test_size": test_size,
    "total_tasks": len(tasks),
    "successful_tasks": len(results),
    "failed_tasks": len(failures),
    "summary_statistics": summary
}
with open(f"results/{filename_prefix}_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# Optionally also save everything in a single file
with open(f"results/{filename_prefix}_full_output.json", "w") as f:
    json.dump({"metadata": metadata, "results": results, "failures": failures}, f, indent=4)

print("\n📊 Benchmarking complete.")
print(f"✔️ {len(results)} successful, ❌ {len(failures)} failed.")
print(f"📁 Files saved with prefix: {filename_prefix}")