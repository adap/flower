"""Used to test the model and the data partitionning."""

from flwr_baselines.publications.fedavg_mnist import model


def test_cnn_size_mnist() -> None:
    """Test number of parameters with MNIST-sized inputs."""
    # Prepare
    net = model.Net()
    expected = 1_663_370

    # Execute
    actual = sum([p.numel() for p in net.parameters()])

    # Assert
    assert actual == expected
