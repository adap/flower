import numpy as np
from flower.datasets.flwr_datasets.metrics.get_distros import get_distros


class HellingerDistance:
    """Hellinger distance for multiple clients' targets.

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
        """Calculate the Hellinger distance for multiple clients' targets.

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
        from flower.datasets.flwr_datasets.metrics.hellinger_distance import HellingerDistance
        random_targets_lists = list(np.random.randint(low=0, high=20, size=(10, 100)))

        HD = HellingerDistance().calculate(random_targets_lists)
        print(HD)
        """
        distributions = get_distros(targets_per_client, self.task_type, self.num_bins)

        n = len(distributions)
        sqrt_d = np.sqrt(distributions)
        h = np.sum((sqrt_d[:, np.newaxis, :] - sqrt_d[np.newaxis, :, :]) ** 2, axis=2)
        hd_val = np.sqrt(np.sum(h) / (2 * n * (n - 1)))
        hd_val = min(hd_val, 1.0)
        return hd_val
