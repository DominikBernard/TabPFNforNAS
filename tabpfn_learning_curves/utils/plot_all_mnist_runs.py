import matplotlib.pyplot as plt


from tabpfn_learning_curves.utils.log_runs_to_tensors import extract_logs_from_subdirectories, create_tensors_from_logs

sorted_strings = [
    "at=mlp,bs=64",
    "at=mlp,bs=128",
    "at=mlp,bs=256",
    "at=mlp,bs=512",
    "at=mlp,bs=1024",
    "at=cnn,bs=64",
    "at=cnn,bs=128",
    "at=cnn,bs=256",
    "at=cnn,bs=512",
    "at=cnn,bs=1024"
]

# get the data from the log files
path_to_logs = "tabpfn_learning_curves/MNIST_example_runs"
log_names = extract_logs_from_subdirectories(path_to_logs)

# sort the subdirectories according to the string contained in sorted_strings
log_names = [subdir for string in sorted_strings for subdir in log_names if string in subdir]

learning_curve_data_list = create_tensors_from_logs(log_names, acc_in_percent=True)

# Plot the results for visualization
plt.figure(figsize=(10, 6))
# plot training runs
for i in range(len(learning_curve_data_list)):
    plt.plot(learning_curve_data_list[i][:, 0], learning_curve_data_list[i][:, 1], '-', label=f'Běh {i+1}, Architektura: {sorted_strings[i].split(",")[0].split("=")[1].upper()}')

plt.xlabel('Epocha')
plt.ylabel('Přesnost (%)')
plt.legend()

plt.grid()

plt.tight_layout()

plt.show()
# plt.savefig(f"tabpfn_learning_curves/plots/mnist_multiple_runs/ucici_krivky.pdf", dpi=300)
plt.close()

