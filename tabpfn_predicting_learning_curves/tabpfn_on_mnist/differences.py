# in this code, we will compute differences betweeen each point in the learning curve and the previous one
# this will give us the improvement or deterioration at each epoch
# we will then use these differences as the target values for the TabPFNRegressor


import numpy as np
import matplotlib.pyplot as plt

from tabpfn import TabPFNRegressor

from tabpfn_predicting_learning_curves.utils.tensor_extractor import extract_epoch_dev_accuracy

splitting_epoch = 20  # Define the epoch at which to split training and testing data

# Extract learning curve data from log file
log_file_path = "tabpfn_predicting_learning_curves/MNIST_example_runs/run1.txt"
learning_curve_data = extract_epoch_dev_accuracy(log_file_path, acc_in_percent=True)

# Prepare dataset
X = learning_curve_data[:, 0].reshape(-1, 1)  # Epoch numbers as features
y = np.diff(np.hstack([0, learning_curve_data[:, 1]]))  # Differences in accuracy as target values

# # add 10/x as a feature
# X = np.hstack((X, 10 / X))

# add ultimate feature
# it is a target feature, with some noise
# X = np.hstack((X, (y + np.random.normal(0, 0.005, size=y.shape))[:, np.newaxis]))

# add target as a feature
# X = np.hstack((X, y[:, np.newaxis]))

# split the data into training and testing sets
X_train, X_test = X[:splitting_epoch], X[splitting_epoch:]
y_train, y_test = y[:splitting_epoch], y[splitting_epoch:]


# Initialize a regressor
regressor = TabPFNRegressor()
regressor.fit(X_train, y_train)
# Make predictions
y_pred = regressor.predict(X_test)

# measure the mean squared error
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse}")

# # convert differences back to actual values for plotting
# y = np.cumsum(y)  # Cumulative sum to get actual accuracy values
# y_pred = np.cumsum(y_pred) + y[splitting_epoch] # Cumulative sum of predictions
# mse_after_cumsum = np.mean((y_pred - y_test) ** 2)
# print(f"Mean Squared Error after cumulative sum: {mse_after_cumsum}")

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

