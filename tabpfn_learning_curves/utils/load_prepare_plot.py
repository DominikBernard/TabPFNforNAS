import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os
import tqdm

from tabpfn_final_accuracy.utils.dataloader import download_data_if_not_exists

def load_data(path_to_json: str = "tabpfn_final_accuracy/data/merged_data.json") -> dict:
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

def prepare_dataset(data: dict, train_examples: int = 10, test_examples: int = 200, test_splitting_epoch: int = 10, sort_indices: bool = False) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[list[list[int]], list[int]]], dict[int, tuple[list[list[int]], list[int]]], np.ndarray, np.ndarray, tuple[int, int]]:
    """Prepare the dataset for training and testing.
    Args:
        data (dict): The loaded data containing architectures and their accuracies.
        train_examples (int): Number of training examples to sample.
        test_examples (int): Number of test examples to sample.
        test_splitting_epoch (int): Epoch at which to split the target data into train and test sets.
        sort_indices (bool): Whether to sort the indices.
    Returns:
        tuple: A tuple containing training features, training labels, 2 dictionaries of test architectures (test and train), training indices, test indices, and the implicit shape of the test labels.
    """

    test_indices = np.random.choice(len(data), test_examples, replace=False)
    remaining_indices = np.setdiff1d(np.arange(len(data)), test_indices)
    train_indices = np.random.choice(remaining_indices, train_examples, replace=False)

    if sort_indices:
        all_indices = np.concatenate((train_indices, test_indices))
        sorted_indices = np.argsort(np.argsort([data[idx]["accuracies"][str(test_splitting_epoch - 1)]["test_accuracy"] for idx in all_indices]))

    train_features = []
    train_labels = []

    for i, arch_idx in enumerate(train_indices):
        for epoch in range(len(data[arch_idx]["accuracies"])):
            features = [i if not sort_indices else sorted_indices[i], epoch]
            label = data[arch_idx]["accuracies"][str(epoch)]["test_accuracy"]
            train_features.append(features)
            train_labels.append(label)

    test_archs_test: dict[int, tuple[list[list[int]], list[int]]] = dict()
    test_archs_train: dict[int, tuple[list[list[int]], list[int]]] = dict()

    for i, arch_idx in enumerate(test_indices):
        index_feature = i + len(train_indices) if not sort_indices else sorted_indices[i + len(train_indices)]  # Ensure test indices are distinct from train indices
        for epoch in range(len(data[arch_idx]["accuracies"])):
            features = [index_feature, epoch]
            label = data[arch_idx]["accuracies"][str(epoch)]["test_accuracy"]
            if epoch < test_splitting_epoch:
                if arch_idx not in test_archs_train:
                    test_archs_train[arch_idx] = ([], [])
                test_archs_train[arch_idx][0].append(features)
                test_archs_train[arch_idx][1].append(label)
            else:
                if arch_idx not in test_archs_test:
                    test_archs_test[arch_idx] = ([], [])
                test_archs_test[arch_idx][0].append(features)
                test_archs_test[arch_idx][1].append(label)

    test_labels_implicit_shape = (len(test_indices), len(data[0]["accuracies"]) - test_splitting_epoch)

    return np.array(train_features), np.array(train_labels), test_archs_test, test_archs_train, train_indices, test_indices, test_labels_implicit_shape

def calc_results(predictions: np.ndarray, y_test: np.ndarray, implicit_test_labels_shape: tuple) -> tuple:
    """
    Calculate the results (MSE, variances) for the predictions.

    Args:
        predictions (np.ndarray): The predicted values.
        y_test (np.ndarray): The true values.
        implicit_test_labels_shape (tuple): The shape of the implicit test labels.

    Returns:
        tuple: A tuple containing the reshaped predictions, whole MSE, MSE per architecture, MSE for the last epoch, variance of the whole MSE, and variance for the last epoch.
    """
    whole_mse = np.mean((np.array(predictions) - np.array(y_test)) ** 2)

    y_test_reshaped = np.array(y_test).reshape(implicit_test_labels_shape)
    predictions_reshaped = np.array(predictions).reshape(implicit_test_labels_shape)

    mse_per_arch = np.mean((predictions_reshaped - y_test_reshaped) ** 2, axis=1)

    variance_of_whole_mse = np.var(mse_per_arch, ddof=1)

    mse_for_last_epoch = np.mean((predictions_reshaped[:, -1] - y_test_reshaped[:, -1]) ** 2)

    variance_for_last_epoch = np.var((predictions_reshaped[:, -1] - y_test_reshaped[:, -1]) ** 2, ddof=1)

    return predictions_reshaped, whole_mse, mse_per_arch, mse_for_last_epoch, variance_of_whole_mse, variance_for_last_epoch



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

    for pred_index in tqdm.tqdm(range(len(predictions)), desc="Plotting results"):
        plt.figure(figsize=(10, 6))
        
        accuracies = [data[test_indices[pred_index]]["accuracies"][str(epoch)]["test_accuracy"] for epoch in range(len(data[test_indices[pred_index]]["accuracies"]))]
        plt.plot(accuracies, '-', zorder=3,label=f'Arch {test_indices[pred_index]} (Test)')

        predicted_accuracies = predictions[pred_index]
        plt.plot(range(test_splitting_epoch, test_splitting_epoch + len(predicted_accuracies)), predicted_accuracies, '-', zorder=3, label=f'Predicted arch {test_indices[pred_index]} (Test)')
        
        for i in train_indices:
            accuracies = [data[i]["accuracies"][str(epoch)]["test_accuracy"] for epoch in range(len(data[i]["accuracies"]))]
            plt.plot(accuracies, '--', label=f'Arch {i} (Train)', alpha=0.5)
        
        # plot the splitting epoch
        plt.axvline(x=test_splitting_epoch, color='red', linestyle='--', label='Train/Test Split')

        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.title(f'Learning Curves for Architecture {test_indices[pred_index]}')
        plt.legend()

        if path_to_save:
            plt.savefig(os.path.join(path_to_save, f'learning_curve_arch_{test_indices[pred_index]}.png'))
            # Close the plot to avoid more and more plots being opened in memory
            plt.close()
        else:
            plt.show()
            break  # Show only the first plot when not saving

def store_results(test_indices: np.ndarray, predictions: np.ndarray, mse: float, mse_per_arch: np.ndarray, mse_for_last_epoch: float, variance_of_whole_mse: float, variance_for_last_epoch: float, path_to_save: Optional[str] = None):
    """
    Store the results in a txt file if a path is provided, otherwise print the results.

    Args:
        test_indices (np.ndarray): Indices of the test examples.
        predictions (np.ndarray): The predicted values.
        mse (float): Mean squared error on the test set.
        mse_per_arch (np.ndarray): Mean squared error per architecture.
        mse_for_last_epoch (float): Mean squared error for the last epoch.
        variance_of_whole_mse (float): Variance of the whole mean squared error.
        variance_for_last_epoch (float): Variance for the last epoch.
        path_to_save (Optional[str]): Path to save the results. If None, the results will be printed.
    """

    result_str = "Results:\n"
    result_str += f"Whole Mean Squared Error: {mse}\n"
    result_str += f"Variance of Whole Mean Squared Error: {variance_of_whole_mse}\n"
    result_str += f"Mean Squared Error per Architecture:\n"
    for idx, arch_mse in zip(test_indices, mse_per_arch):
        result_str += f"  Arch {idx}: {arch_mse}\n"
    result_str += f"Mean Squared Error for Last Epoch: {mse_for_last_epoch}\n"
    result_str += f"Variance for Last Epoch: {variance_for_last_epoch}\n"

    if path_to_save:
        with open(os.path.join(path_to_save, "results.txt"), 'w') as file:
            file.write(result_str)
        with open(os.path.join(path_to_save, "predictions.txt"), 'w') as file:
            file.write("Test Index, Predicted Accuracies\n")
            for idx, pred in zip(test_indices, predictions):
                file.write(f"{idx}, {pred.tolist()}\n")

    else:
        print(result_str)


