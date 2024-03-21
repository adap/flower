import numpy as np


def get_distros(targets_per_client, task_type, num_bins):
    """Get the distributions (percentages) for multiple clients' targets.

    Parameters
    ----------
    targets_per_client : list of lists, array-like
        Targets (labels) for each client (local node).
    task_type : str
        Type of task represented in the targets. Should be either 'classification' or 'regression'.
    num_bins : int
        Number of bins used to bin the targets when the task is 'regression'.

    Returns
    -------
    pctg_distr: list of lists, array like
        Distributions (percentages) of the clients' targets.
    """
    if task_type not in ["classification", "regression"]:
        raise ValueError(
            "Task '"
            + task_type
            + "' not implemented. Available tasks for non-IID-ness metrics calculation are: ["
            "'classification', "
            "'regression']."
        )
    else:
        # Flatten targets array
        targets = np.concatenate(targets_per_client)

        # Bin target for regression tasks
        if task_type == "regression":
            targets_per_client, targets = bin_targets_per_client(
                targets, targets_per_client, num_bins
            )

        # Get unique classes and counts
        unique_classes, counts = np.unique(targets, return_counts=True)

        # Calculate distribution (percentage) for each client
        pctg_distr = []
        for client_targets in targets_per_client:
            # Count occurrences of each unique class in client's targets
            client_counts = np.bincount(
                np.searchsorted(unique_classes, client_targets),
                minlength=len(unique_classes),
            )
            # Get percentages
            client_perc = client_counts / len(client_targets)
            pctg_distr.append(client_perc.tolist())

    return pctg_distr


def bin_targets(targets, num_bins):
    """Get the target binned.

    Parameters
    ----------
    targets : lists
        Targets (labels) variable.

    num_bins : int
        Number of bins used to bin the targets when the task is 'regression'.

    Returns
    -------
    bins: list
        Bins calculated.
    binned_targets:
        Binned target variable.
    """
    # Compute bins
    bins = np.linspace(min(targets), max(targets), num_bins + 1)
    # Bin the targets
    binned_targets = np.digitize(targets, bins)
    return bins, binned_targets


def bin_targets_per_client(targets, targets_per_client, num_bins):
    """Get the target binned.

    Parameters
    ----------
    targets : lists
        Targets (labels) variable.
    targets_per_client : lists of lists, array-like
        Targets (labels) for each client (local node).
    num_bins : int
        Number of bins used to bin the targets when the task is 'regression'.

    Returns
    -------
    binned_targets_per_client: list
        Bins calculated target of each client.
    binned_targets:
        Binned target variable.
    """
    # Bin targets
    bins, binned_targets = bin_targets(targets, num_bins)
    # Bin each clients' target using calculated bins
    binned_targets_per_client = []
    for client_targets in targets_per_client:
        binned_client_targets = list(np.digitize(client_targets, bins))
        binned_targets_per_client.append(binned_client_targets)
    return binned_targets_per_client, binned_targets
