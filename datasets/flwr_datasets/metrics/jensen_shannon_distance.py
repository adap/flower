import math

import numpy as np
from flower.datasets.flwr_datasets.metrics.get_distros import get_distros


class JensenShannonDistance:
    """Jensen-Shannon distance for multiple clients' targets.

    Parameters
    ----------
    task_type : str
        Type of task represented in the targets. Should be either 'classification' or 'regression'.
    num_bins : int
        Number of bins used to bin the targets when the task is 'regression'.
    """

    def __init__(self, task_type="classification", num_bins=20):
        self.task_type = task_type
        self.num_bins = num_bins

    def calculate(self, targets_per_client):
        """Calculate the Jensen-Shannon distance for multiple clients' targets.

        Parameters
        ----------
        targets_per_client : list of lists, array-like
            Targets (labels) for each client (local node).

        Returns
        -------
        hd_val: float
            Hellinger distance.

        Examples
        --------
        import numpy as np
        from flower.datasets.flwr_datasets.metrics.hellinger_distance import JensenShannonDistance
        random_targets_lists = list(np.random.randint(low=0, high=20, size=(10, 100)))

        JSD = JensenShannonDistance().calculate(random_targets_lists)
        print(JSD)
        """
        distributions = get_distros(targets_per_client, self.task_type, self.num_bins)

        # Set weights to be uniform
        weight = 1 / len(distributions)
        js_left = np.zeros(len(distributions[0]))
        js_right = 0
        for prd in distributions:
            js_left += np.array(prd) * weight
            js_right += weight * entropy(prd, normalize=False)

        jsd_val = entropy(js_left, normalize=False) - js_right

        if len(distributions) > 2:
            jsd_val = normalize_value(
                jsd_val, min_val=0, max_val=math.log2(len(distributions))
            )

        jsd_val = min(np.sqrt(jsd_val), 1.0)
        return jsd_val


def entropy(prob_dist, normalize=True):
    """Calculate the entropy.

    Parameters
    ----------
    prob_dist : array-like
        Distribution (percentages) of targets for each local node (client).
    normalize : bool
        Flag to normalize the entropy.

    Returns
    -------
    entropy_val: float
        Entropy.
    """
    entropy_val = -sum([p * math.log2(p) for p in prob_dist if p != 0])
    if normalize:
        max_entropy = math.log2(prob_dist.shape[0])
        return entropy_val / max_entropy
    return entropy_val


def normalize_value(value, min_val=0, max_val=1):
    """Scale (Normalize) input value between min_val and max_val.

    Parameters
    ----------
    value : float
        Value to be normalized.
    min_val : float
        Minimum bound of normalization.
    max_val : float
        Maximum bound of normalization.

    Returns
    -------
    val_norm: float
        Normalized value between min_val and max_val.
    """
    val_norm = (value - min_val) / (max_val - min_val)
    return val_norm
