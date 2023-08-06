"""Used to test the model and the data partitionning."""
# pylint: disable=E0401 disable=R1728

from flwr_baselines.publications.fedavg.cifar10 import model


def test_cnn_size_mnist() -> None:
    """Test number of parameters with CIFAR10-sized inputs."""
    # Prepare
    net = model.Net()
    expected = 615_338

    # Execute
    actual = sum([p.numel() for p in net.parameters()])

    # Assert
    assert actual == expected
