import numpy as np
import argparse
from typing import Literal
import nats_bench
import gdown
import os

from tabpfn_with_nats_bench.utils.arch_features import get_features_and_label, get_basic_features, is_arch_connected


def create_dataset_from_data(indices_to_choose_from: np.ndarray, args: argparse.Namespace, data: list[dict], test_or_train: Literal["test", "train"] = "train") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create dataset of features and labels.
    Args:
        indices_to_choose_from: Indices to choose architectures from.
        args: Command line arguments.
        test_or_train: Whether to create test or train dataset.

    Returns:
        chosen_indices: Indices of chosen architectures.
        features: Features of chosen architectures.
        labels: Labels of chosen architectures.
    """
    size = args.test_size if test_or_train == "test" else args.train_size
    chosen_indices = np.random.choice(indices_to_choose_from, size, replace=False)

    features_and_labels = [get_features_and_label(data, int(i), args.connected_feature, args.arch_index_feature, args.edge_features, args.zero_cost_features, args.epochs_n_features) for i in chosen_indices]
    features, labels = zip(*features_and_labels)

    features = np.array(features)
    labels = np.array(labels)

    return chosen_indices, features, labels

def create_dataset_from_api(indices_to_choose_from: np.ndarray, args: argparse.Namespace, api: nats_bench.NATStopology, test_or_train: Literal["test", "train"] = "train") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create dataset of features and labels.
    Args:
        indices_to_choose_from: Indices to choose architectures from.
        args: Command line arguments.
        test_or_train: Whether to create test or train dataset.

    Returns:
        chosen_indices: Indices of chosen architectures.
        features: Features of chosen architectures.
        labels: Labels of chosen architectures.
    """
    size = args.test_size if test_or_train == "test" else args.train_size
    chosen_indices = np.random.choice(indices_to_choose_from, size, replace=False)

    archs = [get_basic_features(api.arch(int(i)), i, args.connected_feature, args.arch_index_feature, args.edge_features_onehot) for i in chosen_indices]
    features = np.array(archs)

    labels = [api.get_more_info(int(i), args.dataset, hp="200", is_random=False)['test-accuracy'] for i in chosen_indices]
    labels = np.array(labels)

    return chosen_indices, features, labels

def skip_unconnected_archs_using_data(data: list[dict], indices: np.ndarray) -> np.ndarray:
    """Filter out unconnected architectures from the given indices."""
    connected_indices = []
    for i in indices:
        arch = data[i]["arch"]
        if is_arch_connected(arch):
            connected_indices.append(i)
    return np.array(connected_indices)

def download_data_if_not_exists(datafile: str, url: str = 'https://drive.google.com/uc?id=1htCa-ztKlkkYyje6K98q8n-CioPUA_Pk'):
    """Download data from the given URL if the data file does not exist."""
    if not os.path.exists(datafile):
        print(f"Data file {datafile} does not exist, downloading it from {url}...")

        # create the directory if it does not exist
        if not os.path.exists(os.path.dirname(datafile)):
            os.makedirs(os.path.dirname(datafile), exist_ok=True)

        # download them from google drive
        gdown.download(url, datafile, quiet=False)

def load_predictions_file(file_path):
    """Load a file and return its contents as two numpy arrays (vectors -- indices and predictions)."""

    # check if the file exists
    if not os.path.isfile(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    # load the file and return the contents as two numpy arrays
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    indices = data[:, 0].astype(np.int64)
    predictions = data[:, 1]
    return indices, predictions

def get_true_labels(indices, data):
    """Get the true labels for the given indices from the data."""
    true_labels = []
    for i in indices:
        arch_data = data[int(i)]
        true_label = arch_data["testacc"]
        true_labels.append(true_label)
    return np.array(true_labels)
