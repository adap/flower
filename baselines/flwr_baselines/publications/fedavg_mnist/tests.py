"""Used to test the model and the data partitionning."""


from flwr_baselines.publications.fedavg_mnist import dataset, model


def test_cnn_size_mnist() -> None:
    """Test number of parameters with MNIST-sized inputs."""
    # Prepare
    net = model.Net()
    expected = 1_663_370

    # Execute
    actual = sum([p.numel() for p in net.parameters()])

    # Assert
    assert actual == expected


def test_non_iid_partitionning(num_clients: int = 100) -> None:
    """Test the non iid partitionning of the MNIST dataset.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients to distribute the data to, by default 100
    """
    trainloaders, _, _ = dataset.load_datasets(
        batch_size=1, num_clients=num_clients, iid=False
    )
    for trainloader in trainloaders:
        labels = []
        for _, label in trainloader:
            labels.append(label.item())
        assert len(set(labels)) <= 2
