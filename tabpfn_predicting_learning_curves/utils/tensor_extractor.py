import re
import numpy as np


def extract_epoch_dev_accuracy(file_path: str, acc_in_percent: bool = False) -> np.ndarray:
    """
    Extracts epoch numbers and dev_accuracy values from a log file and returns them as a NumPy array.

    Args:
        file_path (str): Path to the log file.
        acc_in_percent (bool): Whether to convert accuracy to percentage.

    Returns:
        np.ndarray: A 2D array where the first column is epoch numbers and the second column is dev_accuracy values.
    """
    # Read the log file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    epochs = []
    dev_accuracies = []

    # Extract epoch number and dev_accuracy using regex
    pattern = r"Epoch (\d+)/\d+.*dev_accuracy=(\d+\.\d+)"
    for line in lines:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            dev_acc = float(match.group(2))
            epochs.append(epoch)
            dev_accuracies.append(dev_acc)

    # Create the tensor (NumPy array)
    tensor = np.column_stack((epochs, dev_accuracies))
    if acc_in_percent:
        tensor[:, 1] *= 100  # Convert accuracy to percentage if specified
    return tensor


if __name__ == "__main__":
    file_path = "tabpfn_predicting_learning_curves/MNIST_example_runs/run1.txt"
    tensor = extract_epoch_dev_accuracy(file_path)
    print(tensor)