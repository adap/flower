"""Used to test the model and the data partitionning."""


from flwr_baselines.publications.fedavg_mnist_2nn import model


def test_2nn_size_mnist() -> None:
    """Test number of parameters with MNIST-sized inputs."""
    # Prepare
    net = model.Net()
    expected = 199_210

    # Execute
    actual = sum([p.numel() for p in net.parameters()])

    # Assert
    assert actual == expected
