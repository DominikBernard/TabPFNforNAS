import numpy as np
import matplotlib.pyplot as plt
import argparse

from tabpfn import TabPFNRegressor

import tabpfn_learning_curves.utils.load_prepare_plot as lpp


parser = argparse.ArgumentParser(description="Predict learning curves using TabPFN on NATS-Bench data.")
parser.add_argument("--splitting_epoch", type=int, default=20, help="Epoch at which to split the target data into train and test sets.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--datapath", type=str, default="tabpfn_with_nats_bench/data/merged_data.json", help="Path to the JSON file containing the data.")

def load_arch_data(data: dict, arch_idx: int) -> np.ndarray:
    """Load architecture data and prepare features and labels.
    Args:
        data (dict): The loaded data containing architectures and their accuracies.
        arch_idx (int): The index of the architecture to load.
    Returns:
        np.ndarray: An array of features and labels for the specified architecture.
    """
    
    learning_curve_data = []
    for epoch in range(len(data[arch_idx]["accuracies"])):
        feature = epoch
        label = data[arch_idx]["accuracies"][str(epoch)]["test_accuracy"]
        learning_curve_data.append((feature, label))

    return np.array(learning_curve_data)

def prepare_X_y(learning_curve_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prepare feature and label arrays from the learning curve data.
    Args:
        learning_curve_data (np.ndarray): An array of features and labels for a specific architecture.
    Returns:
        tuple: A tuple containing the feature array X and the label array y.
    """
    X = learning_curve_data[:, 0].reshape(-1, 1)  # Epoch numbers as features
    y = learning_curve_data[:, 1]  # Dev accuracy as target values

    # # add sine and cosine features to X
    # X = np.hstack((X, 100 * np.sin(X[:, 0])[:, np.newaxis]))
    # X = np.hstack((X, 100 * np.cos(X[:, 0])[:, np.newaxis]))
    # X = np.hstack((X, 100 * np.sin(X[:, 0] * np.pi / 200)[:, np.newaxis]))
    # X = np.hstack((X, 100 * np.cos(X[:, 0] * np.pi / 200)[:, np.newaxis]))

    # # add 1-1/x feature to X
    # X = np.hstack((X, 100 * (0.9 - 1 / (X[:, 0] + 1))[:, np.newaxis]))

    # # add log feature to X
    # X = np.hstack((X, 100 * ((np.log(X[:, 0] + 1))/3)[:, np.newaxis]))

    # # add square feature to X
    # X = np.hstack((X, (X[:, 0] ** 2 / 10_000)[:, np.newaxis]))

    # # add ultimate feature
    # X = np.hstack((X, 100 * (np.sin(X[:, 0] * np.pi / 200) + 2 * (1 - 3 / (X[:, 0] + 1)) / 3)[:, np.newaxis]))

    # add ulitimate feature number 2
    # it is a combination feature 4 until step 9 and feature 3 after step 10
    # X = np.hstack((X, 100 * (np.where(X[:, 0] < 10, ((np.log(X[:, 0] + 1))/3), (0.9 - 1 / (X[:, 0] + 1))))[:, np.newaxis]))

    # add ultimate feature number 3
    # it is a target feature, with some noise
    # X = np.hstack((X, (y + np.random.normal(0, 0.5, size=y.shape))[:, np.newaxis]))

    # add target as a feature
    # X = np.hstack((X, y[:, np.newaxis]))

    return X, y

def split_train_test(X: np.ndarray, y: np.ndarray, splitting_epoch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into training and testing sets based on the specified splitting epoch.
    Args:
        X (np.ndarray): The feature array.
        y (np.ndarray): The label array.
        splitting_epoch (int): The epoch at which to split the data into training and testing sets.
    Returns:
        tuple: A tuple containing the training features, training labels, testing features, and testing labels.
    """

    # split the data into training and testing sets
    X_train, X_test = X[:splitting_epoch], X[splitting_epoch:]
    y_train, y_test = y[:splitting_epoch], y[splitting_epoch:]

    return X_train, y_train, X_test, y_test

def make_predictions(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Train the TabPFNRegressor and make predictions on the test set.
    Args:
        X_train (np.ndarray): The training feature array.
        y_train (np.ndarray): The training label array.
        X_test (np.ndarray): The testing feature array.
    Returns:
        np.ndarray: The predicted labels for the test set.
    """

    # Initialize a regressor
    regressor = TabPFNRegressor()
    regressor.fit(X_train, y_train)
    # Make predictions
    y_pred = regressor.predict(X_test)

    return y_pred

def calc_mse(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Calculate the mean squared error between the predicted and actual labels.
    Args:
        y_pred (np.ndarray): The predicted labels.
        y_test (np.ndarray): The actual labels.
    Returns:
        float: The mean squared error.
    """
    # measure the mean squared error
    mse = np.mean((y_pred - y_test) ** 2)

    return mse

def plot_results(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_pred: np.ndarray, splitting_epoch: int):
    """Plot the actual and predicted learning curves with a vertical line indicating the train/test split.
    Args:
        X (np.ndarray): The feature array for the entire dataset.
        y (np.ndarray): The label array for the entire dataset.
        X_test (np.ndarray): The feature array for the test set.
        y_pred (np.ndarray): The predicted labels for the test set.
        splitting_epoch (int): The epoch at which the train/test split occurs.
    """

    # Plot the results for visualization
    # with a straight line splitting training and testing data
    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 0], y, 'o-', label='Actual Dev Accuracy')

    if X.shape[1] > 1:
        for i in range(1, X.shape[1]):
            plt.plot(X[:, 0], X[:, i], 'x--', label=f'Feature {i}', alpha=0.5)

    plt.plot(X_test[:, 0], y_pred, 'o--', label='Predicted Dev Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Dev Accuracy')
    plt.title('Learning Curve')
    plt.axvline(x=splitting_epoch, color='red', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()

    # fix random seed
    np.random.seed(args.seed)

    data = lpp.load_data(args.datapath)

    arch_idx = np.random.choice(len(data))

    learning_curve_data = load_arch_data(data, arch_idx)

    X, y = prepare_X_y(learning_curve_data)
    X_train, y_train, X_test, y_test = split_train_test(X, y, args.splitting_epoch)

    y_pred = make_predictions(X_train, y_train, X_test)

    mse = calc_mse(y_pred, y_test)
    print(f"Mean Squared Error: {mse}")

    plot_results(X, y, X_test, y_pred, args.splitting_epoch)

