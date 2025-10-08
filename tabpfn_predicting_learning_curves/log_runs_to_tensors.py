import numpy as np
from tensor_extractor import extract_epoch_dev_accuracy
import os

def extract_logs_from_subdirectories(directory: str) -> list[str]:
    """
    Extract all logs.txt files from subdirectories of a given directory.
    Args:
        directory (str): Path to the main directory.
    Returns:
        list[str]: List of paths to logs.txt files.
    """

    log_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename == "logs.txt":
                log_files.append(os.path.join(root, filename))
    return log_files

def create_tensors_from_logs(log_files: list[str]) -> list[np.ndarray]:
    """
    Create tensors from a list of log files.
    Args:
        log_files (list[str]): List of paths to log files.

    Returns:
        list[np.ndarray]: List of tensors created from log files.
    """
    tensors = []
    for log_file in log_files:
        data = extract_epoch_dev_accuracy(log_file)
        tensor = np.array(data)
        tensors.append(tensor)
    return tensors

if __name__ == "__main__":
    main_directory = ".my_work/tabpfn_predicting_learning_curves/logs"  # Replace with your main directory path
    log_files = extract_logs_from_subdirectories(main_directory)
    tensors = create_tensors_from_logs(log_files)
    
    for i, tensor in enumerate(tensors):
        print(f"Tensor from log file {log_files[i]}:\n{tensor}\n")