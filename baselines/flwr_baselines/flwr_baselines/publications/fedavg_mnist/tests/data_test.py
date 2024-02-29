"""Used to test the model and the data partitionning."""

from flwr_baselines.publications.fedavg_mnist import dataset


def test_non_iid_partitionning(num_clients: int = 100) -> None:
    """Test the non iid partitionning of the MNIST dataset.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients to distribute the data to, by default 100
    """
    trainloaders, _, _ = dataset.load_datasets(
        batch_size=1, num_clients=num_clients, iid=False, balance=True
    )
    for trainloader in trainloaders:
        labels = []
        for _, label in trainloader:
            labels.append(label.item())
        assert len(set(labels)) <= 2
