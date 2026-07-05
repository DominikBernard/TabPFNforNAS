import numpy as np
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor

from tabpfn_learning_curves.utils.tensor_extractor import extract_epoch_dev_accuracy

# constants
SCALING_FACTOR = 100  # Scaling factor for features to keep them in a similar range as the target values (accuracy in percent)

def load_learning_curve_data(log_file_path):
    """Extracts learning curve data into X (epochs) and y (accuracy)."""
    data = extract_epoch_dev_accuracy(log_file_path, acc_in_percent=True)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return X, y

def train_and_predict(X_train, y_train, X_test):
    """Trains a TabPFNRegressor on the training data and predicts on the test data."""
    regressor = TabPFNRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return y_pred

def compute_metrics(y_test, y_pred):
    """Computes mean squared error and squared error of the last prediction."""
    mse = np.mean((y_pred - y_test) ** 2)
    last_pred_squared_error = (y_pred[-1] - y_test[-1]) ** 2
    return mse, last_pred_squared_error

def split_data_compute_predictions_and_metrics(X, y, splitting_epoch):
    """
    Splits the dataset into training and testing sets based on the specified splitting epoch,
    trains a TabPFNRegressor on the training set,
    makes predictions on the testing set, and computes the mean squared error (MSE) and
    the squared error of the last prediction.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target values.
        splitting_epoch (int): The epoch at which to split the data into training and testing sets
    Returns:
        tuple: A tuple containing:
            - y_pred (np.ndarray): The predicted values for the testing set.
            - mse (float): The mean squared error of the predictions on the testing set.
            - last_pred_squared_error (float): The squared error of the last prediction in the testing set
    """

    # split into train and test sets
    X_train, X_test = X[:splitting_epoch], X[splitting_epoch:]
    y_train, y_test = y[:splitting_epoch], y[splitting_epoch:]

    # train and predict
    y_pred = train_and_predict(X_train, y_train, X_test)

    # compute metrics
    mse, last_pred_squared_error = compute_metrics(y_test, y_pred)

    return y_pred, mse, last_pred_squared_error

def plot_learning_curve(epochs, X, y, y_pred, splitting_epoch, ax=None, mse=None, feature_names=None, path_to_save=None):
    """
    Utility function to plot the learning curve along with any engineered features and predictions.

    Plots the actual dev accuracy, any engineered features, and the predicted dev accuracy on the same graph.

    Args:
        epochs (np.ndarray): Array of epoch numbers.
        X (np.ndarray): Training features used for prediction (including epoch numbers and any engineered features).
        y (np.ndarray): Accuracy values.
        y_pred (np.ndarray): Predicted dev accuracy values.
        splitting_epoch (int): Epoch at which the train/test split occurs.
        mse (float, optional): Mean squared error for the predictions.
        feature_names (list of str, optional): Names of the engineered features for labeling the plot.
        path_to_save (str, optional): Path to save the plot.
        ax (matplotlib.axes.Axes, optional): If provided, the plot will be drawn on this axes instead of creating a new figure.

    """
    # plot everything on ax if provided, otherwise create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    ax.plot(epochs, y, '-', label="Skutečná přesnost")
    ax.plot(epochs[splitting_epoch:], y_pred, '-', label='Předpovězená přesnost')

    # plot any engineered features we added
    if X.shape[1] > 1:
        for i in range(1, X.shape[1]):
            if feature_names:
                ax.plot(epochs, X[:, i], 'x--', label=f'Příznak {i}: {feature_names[i-1]}', alpha=0.5)
            else:
                ax.plot(epochs, X[:, i], '--', alpha=0.5, label=('Přidaný příznak' if X.shape[1] == 2 else None))

    ax.axvline(x=splitting_epoch, color='red', linestyle='--', label=f'Dělící epocha: {splitting_epoch}')
    ax.set_xlabel('Epocha')
    ax.set_xticks(np.arange(0, 101, 10))
    ax.grid()

    if mse is not None:
        # display mse under the plot
        ax.text(0.5, -0.15, f'Střední čtvercová chyba: {mse:.4f}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    if standalone:
        ax.set_ylabel('Přesnost (%)')
        ax.set_title('Učící křivka')
        ax.legend()
        plt.tight_layout()
        if path_to_save:
            plt.savefig(path_to_save, dpi=300)
        else:
            plt.show()


def plot_dual_learning_curves(epochs, X1, y1, y_pred1, X2, y2, y_pred2, splitting_epoch, feature_names=None, path_to_save=None):
    """
    Plots two learning curves side by side for comparison.

    Args:
        epochs (np.ndarray): Array of epoch numbers.
        X1 (np.ndarray): Features for the first learning curve.
        y1 (np.ndarray): Actual values for the first learning curve.
        y_pred1 (np.ndarray): Predicted values for the first learning curve.
        X2 (np.ndarray): Features for the second learning curve.
        y2 (np.ndarray): Actual values for the second learning curve.
        y_pred2 (np.ndarray): Predicted values for the second learning curve.
        splitting_epoch (int): Epoch at which the train/test split occurs.
        feature_names (list of str, optional): Names of the engineered features for labeling the plot.
        path_to_save (str, optional): Path to save the plot.
    """
    modify_plotting_parameters()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    plot_learning_curve(epochs, X1, y1, y_pred1, splitting_epoch, ax=axs[0], feature_names=feature_names)
    plot_learning_curve(epochs, X2, y2, y_pred2, splitting_epoch, ax=axs[1], feature_names=feature_names)

    axs[0].set_ylabel('Přesnost (%)')
    axs[1].legend()#loc='lower right')

    axs[0].set_title('Před zpětnou transformací')
    axs[1].set_title('Po zpětné transformaci')

    plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save, dpi=300)
    else:
        plt.show()

def modify_plotting_parameters():
    plt.rcParams.update({
        "font.size": 12,        # base size
        "axes.titlesize": 14,   # title
        "axes.labelsize": 12,   # x and y labels
        "legend.fontsize": 8,  # legend
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

def prepare_multi_run_data(learning_curve_data_list, splitting_epoch, test_run_index=-1, custom_indices=None):
    """
    Prepares data for multiple runs of the learning curve prediction experiment.

    Args:
        learning_curve_data_list (list of np.ndarray): List of learning curve data arrays for each run.
        splitting_epoch (int): Epoch at which to split the data into training and testing sets.
        test_run_index (int, optional): Index of the run to be used as the test run. Defaults to -1 (last run).
        custom_indices (list of int, optional): Custom indices to use for distinguishing runs. If None, default indices will be used.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Feature matrix for training.
            - y_train (np.ndarray): Target values for training.
            - X_test (np.ndarray): Feature matrix for testing.
            - y_test (np.ndarray): Target values for testing.
    """

    X_list = []
    y_list = []
    if custom_indices is None:
        custom_indices = np.arange(len(learning_curve_data_list))

    for i, learning_curve_data in enumerate(learning_curve_data_list):
        X_i = learning_curve_data[:, 0].reshape(-1, 1)  # Epoch numbers as features
        y_i = learning_curve_data[:, 1]  # Dev accuracy as target values

        # add custom index as a first feature to distinguish runs
        X_i = np.hstack((np.full((X_i.shape[0], 1), custom_indices[i]), X_i))

        X_list.append(X_i)
        y_list.append(y_i)

    # handle negative test_run_index
    if test_run_index < 0:
        test_run_index += len(learning_curve_data_list)

    # split the data into training and testing sets
    X_train = np.vstack(X_list[:test_run_index] + X_list[test_run_index + 1:] + [X_list[test_run_index][:splitting_epoch]])
    X_test = X_list[test_run_index][splitting_epoch:]

    y_train = np.hstack(y_list[:test_run_index] + y_list[test_run_index + 1:] + [y_list[test_run_index][:splitting_epoch]])
    y_test = y_list[test_run_index][splitting_epoch:]

    return X_train, y_train, X_test, y_test

def plot_multi_run_learning_curves(learning_curve_data_list, y_pred_list, splitting_epoch, test_run_index=-1, ax=None, path_to_save=None):
    modify_plotting_parameters()

    standalone = False
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 6))
        standalone = True

    # handle negative test_run_index
    if test_run_index < 0:
        test_run_index += len(learning_curve_data_list)

    # plot test run
    test_run_data = learning_curve_data_list[test_run_index]
    ax.plot(test_run_data[:, 0], test_run_data[:, 1], linestyle='-', zorder=3, label=f'Testovací běh')
    # plot predicted values for the test run
    ax.plot(test_run_data[splitting_epoch:, 0], y_pred_list, linestyle='-', zorder=3, label='Předpovězená přesnost (testovacího běhu)')

    # plot training runs
    label = 'Trénovací běh' if len(learning_curve_data_list) == 2 else None
    for i, learning_curve_data in enumerate(learning_curve_data_list):
        if i != test_run_index:
            ax.plot(learning_curve_data[:, 0], learning_curve_data[:, 1], '--', label=label, alpha=0.5)

    if label is None:
        ax.plot([], [], '--', color='gray', label='Trénovací běhy')  # Add a dummy line for the legend
    
    ax.axvline(x=splitting_epoch, color='r', linestyle='--', label='Dělící epocha')
    ax.set_xlabel('Epocha')
    ax.set_ylabel('Přesnost (%)')
    ax.set_title('Učící křivky pro více běhů')
    ax.grid()
    ax.set_xticks(np.arange(0, 101, 10))
    # ax.set_yticks(np.arange(10, 101, 10))

    if standalone:
        ax.legend()
        plt.tight_layout()
        if path_to_save:
            plt.savefig(path_to_save, dpi=300)
        else:
            plt.show()

def plot_multi_run_learning_curves_side_by_side(learning_curve_data_list, y_pred_lists, splitting_epoch, path_to_save=None):
    """
    Plots multiple learning curves side by side for comparison.

    Args:
        learning_curve_data_list (list of np.ndarray): List of learning curve data arrays for each run.
        y_pred_lists (list of np.ndarray): List of predicted values for each run.
        splitting_epoch (int): Epoch at which the train/test split occurs.
        path_to_save (str, optional): Path to save the plot. If not provided, the plot will be displayed.
    """
    
    num_runs = len(learning_curve_data_list)
    num_cols = num_runs // 2 + num_runs % 2  # Calculate number of columns needed for subplots
    fig, axs = plt.subplots(num_cols, 2, figsize=(10, 3 * num_cols))

    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for i in range(num_runs):
        plot_multi_run_learning_curves(learning_curve_data_list, y_pred_lists[i], splitting_epoch, test_run_index=i, ax=axs[i])
        axs[i].set_title(f'Testovaná architektura {"MLP" if i < 5 else "CNN"} (č. {i + 1})')
        if i % 2 == 0:
            axs[i].set_ylabel('Přesnost (%)')
        else:
            axs[i].set_ylabel('')
        if i == num_runs - 1:
            axs[i].legend(loc='lower right')

    plt.tight_layout()

    if path_to_save:
        plt.savefig(path_to_save, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    data_log_file_path = "tabpfn_learning_curves/MNIST_example_runs/at=mlp,bs=256,cs=,d=cosine,e=100,hls=32,lr=0.001,lrf=0.0005,m=0.005,o=SGD,s=42,i=2/logs.txt"
    X, y = load_learning_curve_data(data_log_file_path)
    splitting_epoch = 20
    y_pred, mse, last_pred_squared_error = split_data_compute_predictions_and_metrics(X, y, splitting_epoch)
    plot_learning_curve(X[:, 0], X, y, y_pred, splitting_epoch, mse=mse)


