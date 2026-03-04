import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """Simple 1D CNN for multi-output regression on (n_features, 1) inputs."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        output_dim: int,
        conv1_filters: int = 64,
        conv2_filters: int = 64,
        kernel_size: int = 5,
        use_pooling: bool = False,
    ):
        super().__init__()
        length, channels = input_shape
        pad = (kernel_size - 1) // 2
        self.use_pooling = use_pooling

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=conv1_filters,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=kernel_size,
            padding=pad,
        )

        if use_pooling:
            self.pool = nn.MaxPool1d(kernel_size=2)

        # Compute flattened size
        length_after = length
        if use_pooling:
            length_after = length_after // 2  # after pool1
            length_after = length_after // 2  # after pool2

        self.fc1 = nn.Linear(conv2_filters * length_after, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, T) or (B, T, C). Convert to (B, C, T).
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T, 1)
        x = x.permute(0, 2, 1)  # (B, C, T)

        x = F.relu(self.conv1(x))
        if self.use_pooling:
            x = self.pool(x)

        x = F.relu(self.conv2(x))
        if self.use_pooling:
            x = self.pool(x)

        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_model(
    input_shape: tuple[int, int],
    output_dim: int,
    conv1_filters: int = 64,
    conv2_filters: int = 64,
    kernel_size: int = 5,
    use_pooling: bool = False,
) -> CNN1D:
    return CNN1D(
        input_shape=input_shape,
        output_dim=output_dim,
        conv1_filters=conv1_filters,
        conv2_filters=conv2_filters,
        kernel_size=kernel_size,
        use_pooling=use_pooling,
    )