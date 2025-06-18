#!/usr/bin/env python3
import argparse
import json
import openml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from tabpfn import TabPFNClassifier
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--suite_id", type=int, default=99, help="OpenML benchmark suite ID")
parser.add_argument("--random_state", type=int, default=42, help="Random state for splitting the data")
parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for k-fold cross validation")


def evaluate_task(
    task_id: int, k_folds: int, random_state: int, task_order: tuple[int, int]
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Evaluate a single task from the OpenML benchmark suite using TabPFNClassifier with k-fold CV.
    
    Args:
        task_id (int): OpenML task ID.
        k_folds (int): Number of folds for cross validation.
        random_state (int): Random state for reproducibility.
        task_order (tuple): Current task position and total tasks.
    
    Returns:
        tuple: A tuple containing the results and any errors encountered.
    """
    try:
        # Get task and dataset
        task = openml.tasks.get_task(task_id, download_splits=True)
        dataset = task.get_dataset()
        
        # Get data with proper typing
        data_tuple = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
        X = cast(pd.DataFrame, data_tuple[0])
        y = cast(pd.Series, data_tuple[1])

        # Verify data is loaded
        if X.empty or y.empty:
            raise ValueError(f"Failed to load data for task {task_id}")

        # Convert to numpy arrays
        X_array = X.to_numpy()
        y_array = y.to_numpy()

        # Create k-fold cross validator
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        
        # Initialize lists to store fold results
        fold_accuracies = []
        fold_roc_aucs = []
        
        unique_classes = np.unique(y_array)
        n_classes = len(unique_classes)

        # Perform k-fold cross validation
        for train_index, test_index in kf.split(X_array):
            X_train, X_test = X_array[train_index], X_array[test_index]
            y_train, y_test = y_array[train_index], y_array[test_index]

            # Train model
            clf = TabPFNClassifier()
            clf.fit(X_train, y_train)

            # Predict
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)

            # Evaluate
            accuracy = float(accuracy_score(y_test, y_pred))
            fold_accuracies.append(accuracy)
            
            # Handle ROC AUC calculation based on number of classes
            if n_classes == 2:
                roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr"))
            fold_roc_aucs.append(roc_auc)

        # Calculate mean and std for accuracies and ROC AUCs
        accuracy = float(np.mean(fold_accuracies))
        roc_auc = float(np.mean(fold_roc_aucs))
        accuracy_std = float(np.std(fold_accuracies))
        roc_auc_std = float(np.std(fold_roc_aucs))

        # Store results
        result: Dict[str, Any] = {
            "task_id": task_id,
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": n_classes,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "accuracy_std": accuracy_std,
            "roc_auc_std": roc_auc_std,
        }

        print(f"Task {task_order[0]}/{task_order[1]} (id: {task_id}, name: {dataset.name}): Accuracy={accuracy:.4f}, ROC AUC={roc_auc:.4f}")
        return result, None

    except Exception as e:
        print(f"An error occurred with task {task_order[0]}/{task_order[1]} (id: {task_id}): {e}")
        error_info: Dict[str, Any] = {"task_id": task_id, "error": str(e)}
        return None, error_info


def run_benchmark(
    suite_id: int, k_folds: int, random_state: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], openml.study.OpenMLBenchmarkSuite]:
    """
    Run benchmarking on a suite of OpenML tasks using k-fold cross validation.
    
    Args:
        suite_id (int): OpenML benchmark suite ID.
        k_folds (int): Number of folds for cross validation.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        Tuple containing results, failures, and the benchmark suite object.
    """
    benchmark_suite = openml.study.get_suite(suite_id)
    print(f"Running benchmark suite: {benchmark_suite.name} (ID: {suite_id})")
    print(f"Using {k_folds}-fold cross validation")
    tasks = benchmark_suite.tasks

    if not tasks:
        raise ValueError("No tasks found in the benchmark suite.")
    print(f"Found {len(tasks)} tasks in the benchmark suite.")

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    
    for i, task_id in enumerate(tqdm(tasks, desc="Evaluating tasks", unit="task")):
        task_order: tuple[int, int] = (i + 1, len(tasks))
        result, failure = evaluate_task(task_id, k_folds, random_state, task_order)
        if result is not None:
            results.append(result)
        if failure is not None:
            failures.append(failure)

    return results, failures, benchmark_suite


def save_results(
    results: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    benchmark_suite: openml.study.OpenMLBenchmarkSuite,
    suite_id: int,
    k_folds: int,
    random_state: int
) -> None:
    """
    Save benchmarking results to files.
    
    Args:
        results: List of successful task results.
        failures: List of failed task information.
        benchmark_suite: The OpenML benchmark suite object.
        suite_id: The benchmark suite ID.
        k_folds: The number of folds used for cross validation.
        random_state: The random state used for reproducibility.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename_prefix = f"benchmark_{suite_id}_kfold_{timestamp}"

    # Handle potential empty results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"results/{filename_prefix}_results.csv", index=False)
        
        mean_accuracy = float(results_df["accuracy"].mean())
        std_accuracy = float(results_df["accuracy"].std())
        mean_roc_auc = float(results_df["roc_auc"].mean())
        std_roc_auc = float(results_df["roc_auc"].std())
    else:
        mean_accuracy = 0.0
        std_accuracy = 0.0
        mean_roc_auc = 0.0
        std_roc_auc = 0.0

    summary: Dict[str, float] = {
        "overall_mean_accuracy": mean_accuracy,
        "overall_std_accuracy": std_accuracy,
        "overall_mean_roc_auc": mean_roc_auc,
        "overall_std_roc_auc": std_roc_auc,
    }

    assert benchmark_suite.tasks is not None, "Benchmark suite tasks must be defined"

    metadata: Dict[str, Any] = {
        "benchmark_suite_id": suite_id,
        "benchmark_suite_name": benchmark_suite.name,
        "run_timestamp": timestamp,
        "algorithm": "TabPFNClassifier",
        "algorithm_params": "default",
        "sklearn_random_state": random_state,
        "k_folds": k_folds,
        "validation_method": "k-fold cross validation",
        "total_tasks": len(benchmark_suite.tasks),
        "successful_tasks": len(results),
        "failed_tasks": len(failures),
        "summary_statistics": summary
    }

    with open(f"results/{filename_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    with open(f"results/{filename_prefix}_full_output.json", "w") as f:
        json.dump({
            "metadata": metadata, 
            "results": results, 
            "failures": failures
        }, f, indent=4)

    print("\n📊 Benchmarking complete.")
    print(f"✔️ {len(results)} successful, ❌ {len(failures)} failed.")
    print(f"📁 Files saved with prefix: {filename_prefix}")


def main() -> None:
    """Main function to run the benchmark."""
    args = parser.parse_args([] if "__file__" not in globals() else None)
    results, failures, benchmark_suite = run_benchmark(
        args.suite_id, args.k_folds, args.random_state
    )
    save_results(
        results, failures, benchmark_suite, args.suite_id, args.k_folds, args.random_state
    )


if __name__ == "__main__":
    main()
