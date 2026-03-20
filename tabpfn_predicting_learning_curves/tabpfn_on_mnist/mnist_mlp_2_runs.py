import numpy as np
import matplotlib.pyplot as plt

from tabpfn import TabPFNRegressor

from tabpfn_predicting_learning_curves.utils.tensor_extractor import extract_epoch_dev_accuracy

splitting_epoch = 20  # Define the epoch at which to split training and testing data
differences = False  # Whether to use differences in accuracy as target values

# Extract learning curve data from log files
log_file_paths = [
    "tabpfn_predicting_learning_curves/logs/run1.txt",
    "tabpfn_predicting_learning_curves/logs/run2.txt"
]

learning_curve_data_list = [extract_epoch_dev_accuracy(path) for path in log_file_paths]

# Prepare dataset
X_list = []
y_list = []
for i in range(len(learning_curve_data_list)):
    learning_curve_data = learning_curve_data_list[i]
    X_i = learning_curve_data[:, 0].reshape(-1, 1) # Epoch numbers as features
    if differences:
        y_i = np.diff(learning_curve_data[:, 1], prepend=0) # Differences in accuracy as target values
    else:
        y_i = learning_curve_data[:, 1] # Dev accuracy as target values

    # add i as a first feature to distinguish runs
    X_i = np.hstack((np.full((X_i.shape[0], 1), i), X_i))

    X_list.append(X_i)
    y_list.append(y_i)

# split the data into training and testing sets
# last run is split into train and test
X_train = np.vstack(X_list[:-1] + [X_list[-1][:splitting_epoch]])
X_test = X_list[-1][splitting_epoch:]

y_train = np.hstack(y_list[:-1] + [y_list[-1][:splitting_epoch]])
y_test = y_list[-1][splitting_epoch:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Initialize a regressor
regressor = TabPFNRegressor()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# measure the mean squared error
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse}")

# Plot the results for visualization
plt.figure(figsize=(10, 6))
# plot training runs
for i in range(len(learning_curve_data_list) - 1):
    plt.plot(learning_curve_data_list[i][:, 0], learning_curve_data_list[i][:, 1], '--', label=f'Run {i+1} (Train)', alpha=0.5)

# plot test run
plt.plot(learning_curve_data_list[-1][:, 0], learning_curve_data_list[-1][:, 1], 'g--', label=f'Run {len(learning_curve_data_list)} (Test)')

if not differences:
    plt.plot(X_test[:, 1], y_pred, 'r.-', label='Predicted Dev Accuracy (Test)')
else:
    # convert predicted differences back to accuracy
    last_train_accuracy = learning_curve_data_list[-1][splitting_epoch - 1, 1]
    y_pred_cumsum = np.cumsum(y_pred) + last_train_accuracy
    plt.plot(X_test[:, 1], y_pred_cumsum, 'r.-', label='Predicted Dev Accuracy (Test)')

plt.axvline(x=splitting_epoch, color='k', linestyle='--', label='Train/Test Split')
plt.xlabel('Epoch')
plt.ylabel('Dev Accuracy')
plt.title('Dev Accuracy vs Epoch')
plt.legend()

plt.grid()
plt.show()
# plt.savefig("plots/mnist_mlp_2_runs_tabpfn_example2.png")
plt.close()
