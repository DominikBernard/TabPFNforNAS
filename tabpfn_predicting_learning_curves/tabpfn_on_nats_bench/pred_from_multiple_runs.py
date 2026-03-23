import json
from tabpfn import TabPFNRegressor
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os

from tabpfn_with_nats_bench.utils.dataloader import download_data_if_not_exists

parser = argparse.ArgumentParser(description="Predict learning curves using TabPFN on NATS-Bench data.")
parser.add_argument("--train_size", type=int, default=10, help="Number of training examples.")
parser.add_argument("--test_size", type=int, default=200, help="Number of test examples.")
parser.add_argument("--test_splitting_epoch", type=int, default=10, help="Epoch at which to split the target data into train and test sets.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--datapath", type=str, default="tabpfn_with_nats_bench/data/merged_data.json", help="Path to the JSON file containing the data.")
parser.add_argument("--path_to_dir_to_save_results", type=str, default="tabpfn_predicting_learning_curves/logs/results/", help="Path to save the results. If not provided, the results will be displayed.")

def load_data(path_to_json: str) -> dict:
    """
    Load data from a JSON file.
    Args:
        path_to_json (str): Path to the JSON file.
    Returns:
        dict: Loaded data as a dictionary.
    """
    # download data if it does not exist
    download_data_if_not_exists(path_to_json)

    with open(path_to_json, 'r') as file:
        data = json.load(file)
    return data

def prepare_dataset(data: dict, train_examples: int = 10, test_examples: int = 200, test_splitting_epoch: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Prepare the dataset for training and testing.
    Args:
        data (dict): The loaded data containing architectures and their accuracies.
        train_examples (int): Number of training examples to sample.
        test_examples (int): Number of test examples to sample.
        test_splitting_epoch (int): Epoch at which to split the target data into train and test sets.
    Returns:
        tuple: A tuple containing training features, training labels, test features, test labels, training indices, test indices, and the implicit shape of the test labels.
    """

    test_indices = np.random.choice(len(data), test_examples, replace=False)
    remaining_indices = np.setdiff1d(np.arange(len(data)), test_indices)
    train_indices = np.random.choice(remaining_indices, train_examples, replace=False)

    train_features = []
    train_labels = []

    for i, arch_idx in enumerate(train_indices):
        for epoch in range(len(data[arch_idx]["accuracies"])):
            features = [i, epoch]
            label = data[arch_idx]["accuracies"][str(epoch)]["test_accuracy"]
            train_features.append(features)
            train_labels.append(label)

    test_features = []
    test_labels = []

    for i, arch_idx in enumerate(test_indices):
        index_feature = i + len(train_indices)  # Ensure test indices are distinct from train indices
        for epoch in range(len(data[arch_idx]["accuracies"])):
            features = [index_feature, epoch]
            label = data[arch_idx]["accuracies"][str(epoch)]["test_accuracy"]
            if epoch < test_splitting_epoch:
                train_features.append(features)
                train_labels.append(label)
            else:
                test_features.append(features)
                test_labels.append(label)

    test_labels_implicit_shape = (len(test_indices), len(data[0]["accuracies"]) - test_splitting_epoch)

    return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels), train_indices, test_indices, test_labels_implicit_shape

def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, implicit_test_labels_shape: tuple[int, int]) -> tuple[np.ndarray, float, np.ndarray, float]:
    """
    Train the TabPFNRegressor and evaluate its performance.
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        implicit_test_labels_shape (tuple[int, int]): The implicit shape of the test labels for reshaping predictions.

    Returns:
        tuple: A tuple containing the predictions, the mean squared error on the test set, the mean squared error per architecture, and the mean squared error for the last epoch.
    """
    model = TabPFNRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    whole_mse = np.mean((predictions - y_test) ** 2)

    y_test_reshaped = y_test.reshape(implicit_test_labels_shape)
    predictions_reshaped = predictions.reshape(implicit_test_labels_shape)

    mse_per_arch = np.mean((predictions_reshaped - y_test_reshaped) ** 2, axis=1)

    mse_for_last_epoch = np.mean((predictions_reshaped[:, -1] - y_test_reshaped[:, -1]) ** 2)

    return predictions_reshaped, whole_mse, mse_per_arch, mse_for_last_epoch

def plot_results(data: dict, train_indices: np.ndarray, test_indices: np.ndarray, predictions: np.ndarray, test_splitting_epoch: int, path_to_save: Optional[str] = None):
    """
    Plot the learning curves.
    Args:
        data (dict): The loaded data containing architectures and their accuracies.
        train_indices (np.ndarray): Indices of the training examples.
        test_indices (np.ndarray): Indices of the test examples.
        predictions (np.ndarray): The predicted values.
        test_splitting_epoch (int): Epoch at which to split the target data into train and test sets.
        path_to_save (Optional[str]): Path to save the plots. If None, the first plot will be displayed.
    """

    for pred_index in range(len(predictions)):
        plt.figure(figsize=(10, 6))
        
        for i in train_indices:
            accuracies = [data[i]["accuracies"][str(epoch)]["test_accuracy"] for epoch in range(len(data[i]["accuracies"]))]
            plt.plot(accuracies, '--', label=f'Arch {i} (Train)', alpha=0.5)

        accuracies = [data[test_indices[pred_index]]["accuracies"][str(epoch)]["test_accuracy"] for epoch in range(len(data[test_indices[pred_index]]["accuracies"]))]
        plt.plot(accuracies, 'g--', label=f'Arch {test_indices[pred_index]} (Test)')

        predicted_accuracies = predictions[pred_index]
        plt.plot(range(test_splitting_epoch, test_splitting_epoch + len(predicted_accuracies)), predicted_accuracies, 'r-', label=f'Predicted arch {test_indices[pred_index]} (Test)')
        
        # plot the splitting epoch
        plt.axvline(x=test_splitting_epoch, color='r', linestyle='--', label='Train/Test Split')

        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.title(f'Learning Curves for Architecture {test_indices[pred_index]}')
        plt.legend()

        if path_to_save:
            plt.savefig(os.path.join(path_to_save, f'learning_curve_arch_{test_indices[pred_index]}.png'))
        else:
            plt.show()
            break  # Show only the first plot when not saving

def store_results(test_indices: np.ndarray, predictions: np.ndarray, mse: float, mse_per_arch: np.ndarray, mse_for_last_epoch: float, path_to_save: Optional[str] = None):
    """
    Store the results in a txt file if a path is provided, otherwise print the results.

    Args:
        test_indices (np.ndarray): Indices of the test examples.
        predictions (np.ndarray): The predicted values.
        mse (float): Mean squared error on the test set.
        mse_per_arch (np.ndarray): Mean squared error per architecture.
        mse_for_last_epoch (float): Mean squared error for the last epoch.
        path_to_save (Optional[str]): Path to save the results. If None, the results will be printed.
    """

    result_str = "Results:\n"
    result_str += f"Whole Mean Squared Error: {mse}\n"
    result_str += f"Mean Squared Error per Architecture:\n"
    for idx, arch_mse in zip(test_indices, mse_per_arch):
        result_str += f"  Arch {idx}: {arch_mse}\n"
    result_str += f"Mean Squared Error for Last Epoch: {mse_for_last_epoch}\n"

    if path_to_save:
        with open(os.path.join(path_to_save, "results.txt"), 'w') as file:
            file.write(result_str)
        with open(os.path.join(path_to_save, "predictions.txt"), 'w') as file:
            file.write("Test Index, Predicted Accuracies\n")
            for idx, pred in zip(test_indices, predictions):
                file.write(f"{idx}, {pred.tolist()}\n")

    else:
        print(result_str)



if __name__ == "__main__":
    args = parser.parse_args()

    # fix random seed
    np.random.seed(args.seed)

    data = load_data(args.datapath)

    # Prepare dataset
    X_train, y_train, X_test, y_test, train_indices, test_indices, test_labels_implicit_shape = prepare_dataset(data, train_examples=args.train_size, test_examples=args.test_size, test_splitting_epoch=args.test_splitting_epoch)

    # Train and evaluate the model
    predictions, mse, mse_per_arch, mse_for_last_epoch = train_and_evaluate(X_train, y_train, X_test, y_test, test_labels_implicit_shape)

    plot_results(data, train_indices, test_indices, predictions, args.test_splitting_epoch, path_to_save=args.path_to_dir_to_save_results)
    store_results(test_indices, predictions, mse, mse_per_arch, mse_for_last_epoch, path_to_save=args.path_to_dir_to_save_results)


