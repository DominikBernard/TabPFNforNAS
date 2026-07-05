import numpy as np
from tabpfn_learning_curves.utils.tensor_extractor import extract_epoch_dev_accuracy
import os

def extract_logs_from_subdirectories(directory: str) -> list[str]:
    """
    Extract all logs.txt files from subdirectories of a given directory.
    Args:
        directory (str): Path to the main directory.
    Returns:
        list[str]: List of paths to logs.txt files.
    """

    log_files = [""] * 10
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename == "logs.txt":
                index = int(root[-1]) # assuming the last character of the directory name is the run index
                log_files[index] = os.path.join(root, filename)
    return log_files

def create_tensors_from_logs(log_files: list[str], acc_in_percent: bool = False) -> list[np.ndarray]:
    """
    Create tensors from a list of log files.
    Args:
        log_files (list[str]): List of paths to log files.
        acc_in_percent (bool): Whether to convert accuracy to percentage.

    Returns:
        list[np.ndarray]: List of tensors created from log files.
    """
    tensors = []
    for log_file in log_files:
        data = extract_epoch_dev_accuracy(log_file, acc_in_percent=acc_in_percent)
        tensor = np.array(data)
        tensors.append(tensor)
    return tensors

if __name__ == "__main__":
    main_directory = "tabpfn_learning_curves/MNIST_example_runs"
    log_files = extract_logs_from_subdirectories(main_directory)
    tensors = create_tensors_from_logs(log_files)
    
    for i, tensor in enumerate(tensors):
        print(f"Tensor from log file {log_files[i]}:\n{tensor}\n")