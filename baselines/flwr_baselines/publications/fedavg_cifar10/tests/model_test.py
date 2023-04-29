"""Used to test the model and the data partitionning."""

import model


def test_cnn_size_mnist() -> None:
    """Test number of parameters with CIFAR10-sized inputs."""
    # Prepare
    net = model.Net()
    expected = 2_122_186

    # Execute
    actual = sum([p.numel() for p in net.parameters()])

    # Assert
    assert actual == expected
