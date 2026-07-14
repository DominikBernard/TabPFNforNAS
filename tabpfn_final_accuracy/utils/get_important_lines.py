import regex as re


# find used zero-cost features
def find_zero_cost_features(log_content: str):
    """
    Find the zero-cost features used in the experiment from the log content.

    Args:
        log_content: The content of the log file as a string.

    Returns:
        A list of zero-cost features used in the experiment.
    """
    zcf_match = re.search(r"included zero-cost features:\n((?:\s+- .+\n)+)", log_content)
    if zcf_match:
        zcf_lines = zcf_match.group(1).strip().split("\n")
        zcf_list = [line.split(":")[0].strip().lstrip("- ").strip() for line in zcf_lines]
        return zcf_list
    else:
        return []
    
# find line containing "Mean Squared Error:"
def find_mse(log_content: str):
    """
    Find the mean squared error from the log content.

    Args:
        log_content: The content of the log file as a string.
    Returns:
        The mean squared error as a float, or None if not found.
    """
    mse_match = re.search(r"Mean Squared Error:\s*([0-9.]+)", log_content)
    if mse_match:
        mse_value = float(mse_match.group(1))
        return mse_value
    else:
        return None
    
# find line containing "Spearman's Rank Correlation Coefficient:"
def find_spearman(log_content: str):
    """
    Find the Spearman's rank correlation coefficient from the log content.

    Args:
        log_content: The content of the log file as a string.

    Returns:
        The Spearman's rank correlation coefficient as a float, or None if not found.
    """
    spearman_match = re.search(r"Spearman's Rank Correlation Coefficient:\s*([0-9.-]+)", log_content)
    if spearman_match:
        spearman_value = float(spearman_match.group(1))
        return spearman_value
    else:
        return None
    
# find line containing "Kendall's Tau (rank correlation coefficient):"
def find_kendall(log_content: str):
    """
    Find Kendall's tau from the log content.

    Args:
        log_content: The content of the log file as a string.

    Returns:
        The Kendall's tau as a float, or None if not found.
    """
    kendall_match = re.search(r"Kendall's Tau \(rank correlation coefficient\):\s*([0-9.-]+)", log_content)
    if kendall_match:
        kendall_value = float(kendall_match.group(1))
        return kendall_value
    else:
        return None
    
# find line containing "Used epochs_n_features:"
def find_epochs_n_features(log_content: str):
    """
    Find the number of epochs and features used in the experiment from the log content.

    Args:
        log_content: The content of the log file as a string.

    Returns:
        The number of epochs and features as an integer, or None if not found.
    """
    enf_match = re.search(r'epochs_n_features:\s*(\d+)', log_content)
    if enf_match:
        enf_value = int(enf_match.group(1))
        return enf_value
    else:
        return None
