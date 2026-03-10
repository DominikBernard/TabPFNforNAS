import numpy as np
import argparse
from typing import Literal
import nats_bench

from utils.arch_features import get_features_and_label, get_basic_features, is_arch_connected


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

