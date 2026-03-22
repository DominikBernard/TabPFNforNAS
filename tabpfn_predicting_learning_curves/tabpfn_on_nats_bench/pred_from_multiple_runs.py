import json
from tabpfn import TabPFNRegressor
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os

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
    with open(path_to_json, 'r') as file:
        data = json.load(file)
    return data

def prepare_dataset(data: dict, train_examples: int = 10, test_examples: int = 200, test_splitting_epoch: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the dataset for training and testing.
    Args:
        data (dict): The loaded data containing architectures and their accuracies.
        train_examples (int): Number of training examples to sample.
        test_examples (int): Number of test examples to sample.
        test_splitting_epoch (int): Epoch at which to split the target data into train and test sets.
    Returns:
        tuple: A tuple containing training features, training labels, test features, test labels, training indices, and test indices.
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
        for epoch in range(len(data[arch_idx]["accuracies"])):
            features = [i+len(train_indices), epoch]
            label = data[arch_idx]["accuracies"][str(epoch)]["test_accuracy"]
            if epoch < test_splitting_epoch:
                train_features.append(features)
                train_labels.append(label)
            else:
                test_features.append(features)
                test_labels.append(label)

    return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels), train_indices, test_indices
            
def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Train the TabPFNRegressor and evaluate its performance.
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        tuple: A tuple containing the predictions and the mean squared error on the test set.
    """
    model = TabPFNRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Mean Squared Error on test set: {mse}")

    return predictions, mse

def plot_results(data: dict, train_indices: np.ndarray, test_indices: np.ndarray, test_splitting_epoch: int, path_to_save: Optional[str] = None):
    """
    Plot the learning curves for the training and test examples.
    Args:
        data (dict): The loaded data containing architectures and their accuracies.
        train_indices (np.ndarray): Indices of the training examples.
        test_indices (np.ndarray): Indices of the test examples.
        test_splitting_epoch (int): Epoch at which to split the target data into train and test sets.
        path_to_save (Optional[str]): Path to save the plot. If None, the plot will be displayed.
    """
    plt.figure(figsize=(10, 6))
    
    for i in train_indices:
        accuracies = [data[i]["accuracies"][str(epoch)]["test_accuracy"] for epoch in range(len(data[i]["accuracies"]))]
        plt.plot(accuracies, '--', label=f'Arch {i} (Train)', alpha=0.5)

    for i in test_indices:
        accuracies = [data[i]["accuracies"][str(epoch)]["test_accuracy"] for epoch in range(len(data[i]["accuracies"]))]
        plt.plot(accuracies, 'g--', label=f'Arch {i} (Test)')

    # plot the splitting epoch
    plt.axvline(x=test_splitting_epoch, color='r', linestyle='--', label='Train/Test Split')

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Learning Curves for Training and Test Examples')
    plt.legend()

    if path_to_save:
        plt.savefig(path_to_save)
    else:
        plt.show()

def store_results(test_indices: np.ndarray, mse: float, path_to_save: Optional[str] = None):
    """
    Store the results in a txt file if a path is provided, otherwise print the results.

    Args:
        test_indices (np.ndarray): Indices of the test examples.
        mse (float): Mean squared error on the test set.
        path_to_save (Optional[str]): Path to save the results. If None, the results will be printed.
    """

    result_str = f"Test Indices: {test_indices}\nMean Squared Error: {mse}\n"
    
    if path_to_save:
        with open(path_to_save, 'w') as file:
            file.write(result_str)
    else:
        print(result_str)



if __name__ == "__main__":
    args = parser.parse_args()

    # fix random seed
    np.random.seed(args.seed)

    data = load_data(args.datapath)

    # Prepare dataset
    X_train, y_train, X_test, y_test, train_indices, test_indices = prepare_dataset(data, train_examples=args.train_size, test_examples=args.test_size, test_splitting_epoch=args.test_splitting_epoch)

    # Train and evaluate the model
    predictions, mse = train_and_evaluate(X_train, y_train, X_test, y_test)

    # results handling
    if args.path_to_dir_to_save_results:
        os.makedirs(args.path_to_dir_to_save_results, exist_ok=True)
        path_to_save_plot = os.path.join(args.path_to_dir_to_save_results, "learning_curves.png")
        path_to_save_results = os.path.join(args.path_to_dir_to_save_results, "results.txt")
    else:
        path_to_save_plot = None
        path_to_save_results = None

    plot_results(data, train_indices, test_indices, args.test_splitting_epoch, path_to_save=path_to_save_plot)
    store_results(test_indices, mse, path_to_save=path_to_save_results)