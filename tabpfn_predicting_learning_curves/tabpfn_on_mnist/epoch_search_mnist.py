import numpy as np
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor

from tabpfn_predicting_learning_curves.utils.tensor_extractor import extract_epoch_dev_accuracy

def cumulate_plots_for_different_splitting_epochs(epochs, y, y_pred, splitting_epoch, path_to_save=None):
    height = 3
    width = 2
    fig, axs = plt.subplots(height, width, figsize=(10, 8))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i == len(axs) - 1:
            ax.plot(epochs, y, '-', label="Skutečná přesnost")
            ax.plot(epochs[splitting_epoch[i]:], y_pred[i], '-', label='Předpovězená přesnost', color='orange')
            ax.legend()
        else:
            ax.plot(epochs, y, '-')
            ax.plot(epochs[splitting_epoch[i]:], y_pred[i], '-')
        if i % width == 0:
            ax.set_ylabel('Přesnost (%)')
        ax.set_ylim(-50, 100)
        ax.axvline(x=splitting_epoch[i], color='red', linestyle='--')
        ax.set_xlabel('Epocha')
        ax.set_title(f'Dělící epocha: {splitting_epoch[i]}')
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_yticks(np.arange(-50, 101, 25))
        ax.grid()

    plt.tight_layout()

    if path_to_save:
        plt.savefig(path_to_save, dpi=300)
    else:
        plt.show()

def run_multiple_splitting_epochs(path_to_save_plot=None):
    # set up nicer plot style
    plt.rcParams.update({
        "font.size": 16,        # base size
        "axes.titlesize": 18,   # title
        "axes.labelsize": 16,   # x and y labels
        "legend.fontsize": 14,  # legend
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # example of how to run the same code for multiple splitting epochs and cumulate the plots
    log_file_path = "tabpfn_predicting_learning_curves/MNIST_example_runs/run2.txt"
    learning_curve_data = extract_epoch_dev_accuracy(log_file_path, acc_in_percent=True)
    epochs = learning_curve_data[:, 0]
    y = learning_curve_data[:, 1]

    splitting_epochs = [5, 10, 20, 30, 50, 80]
    y_preds = []
    for splitting_epoch in splitting_epochs:
        X = epochs.reshape(-1, 1)
        X_train, X_test = X[:splitting_epoch], X[splitting_epoch:]
        y_train, y_test = y[:splitting_epoch], y[splitting_epoch:]

        regressor = TabPFNRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_preds.append(y_pred)

    cumulate_plots_for_different_splitting_epochs(epochs, y, y_preds, splitting_epochs, path_to_save=path_to_save_plot)

if __name__ == "__main__":
    run_multiple_splitting_epochs(path_to_save_plot="tabpfn_predicting_learning_curves/plots/no_features/multiple_splitting_epochs2.pdf")


