"""Defines Flower Clients."""

from typing import Dict, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import Dataset

from dasha.compressors import (
    IdentityUnbiasedCompressor,
    UnbiasedBaseCompressor,
    decompress,
)
from dasha.models import ClassificationModel


class CompressionClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Abstract base class for compression clients."""

    SEND_FULL_GRADIENT = "send_full_gradient"
    ACCURACY = "accuracy"
    GRADIENT = "gradient"
    SIZE_OF_COMPRESSED_VECTORS = "size_of_compressed_vectors"

    def __init__(
        self,
        function: ClassificationModel,
        dataset: Dataset,
        device: torch.device,
        compressor: Optional[UnbiasedBaseCompressor] = None,
        evaluate_accuracy=False,
        strict_load=True,
    ):  # pylint: disable=too-many-arguments
        self._function = function.to(device)
        self._function.train()
        self._compressor = (
            compressor if compressor is not None else IdentityUnbiasedCompressor()
        )
        self._local_gradient_estimator = None
        self._gradient_estimator = None
        self._momentum = None
        self._evaluate_accuracy = evaluate_accuracy
        self._dataset = dataset
        self._device = device
        self._strict_load = strict_load

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current model."""
        parameters = [
            val.detach().cpu().numpy().flatten()
            for _, val in self._function.named_parameters()
        ]
        return [np.concatenate(parameters)]

    def _set_parameters(self, parameters_input: NDArrays) -> None:
        """Set the parameters."""
        assert len(parameters_input) == 1
        parameters = parameters_input[0]
        self._compressor.set_dim(len(parameters))
        state_dict = {}
        shift = 0
        for k, parameter_layer in self._function.named_parameters():
            numel = parameter_layer.numel()
            parameter = parameters[shift : shift + numel].reshape(parameter_layer.shape)
            state_dict[k] = torch.Tensor(parameter)
            shift += numel
        missing_keys, unexpected_keys = self._function.load_state_dict(
            state_dict, strict=False
        )
        assert len(unexpected_keys) == 0
        if self._strict_load:
            assert len(missing_keys) == 0

    def _get_current_gradients(self):
        """Get current gradients stored in the PyTorch model."""
        return np.concatenate(
            [val.grad.cpu().numpy().flatten() for val in self._function.parameters()]
        )


def _prepare_full_dataset(dataset, device):
    """Convert PyTorch dataset to a tensor."""
    features, targets = dataset[:]
    features = features.to(device)
    targets = targets.to(device)
    return features, targets


class _GradientCompressionClient(CompressionClient):
    def __init__(self, *args, send_gradient=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._send_gradient = send_gradient
        self._features, self._targets = _prepare_full_dataset(
            self._dataset, self._device
        )

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Send either compressed or noncompressed vector based on the config info."""
        if config[self.SEND_FULL_GRADIENT]:
            compressed_gradient = self._gradient_step(parameters)
        else:
            compressed_gradient = self._compression_step(parameters)
        info = {
            self.SIZE_OF_COMPRESSED_VECTORS: self._compressor.num_nonzero_components()
        }
        return (
            compressed_gradient,
            len(self._targets),
            info,
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Calculate metrics."""
        self._set_parameters(parameters)
        loss = self._function(self._features, self._targets)
        metrics = {}
        if self._send_gradient:
            loss.backward()
            gradients = self._get_current_gradients()
            metrics["gradient"] = gradients.astype(np.float32).tobytes()
        if self._evaluate_accuracy:
            accuracy = self._function.accuracy(self._features, self._targets)
            metrics[self.ACCURACY] = accuracy
        return float(loss), len(self._targets), metrics

    def _calculate_gradient(self, parameters: NDArrays):
        """Calculate the gradient of the PyTorch model."""
        self._set_parameters(parameters)
        self._function.zero_grad()
        loss = self._function(self._features, self._targets)
        loss.backward()
        gradients = self._get_current_gradients()
        return gradients

    def _gradient_step(self, parameters: NDArrays):
        raise NotImplementedError()

    def _compression_step(self, parameters: NDArrays):
        raise NotImplementedError()


class _BaseDashaClient(CompressionClient):
    def _get_momentum(self):
        """Calculate omega from Theorem 6.1 in the DASHA paper."""
        if self._momentum is not None:
            return self._momentum
        self._momentum = 1 / (1 + 2 * self._compressor.omega())
        return self._momentum


class DashaClient(_GradientCompressionClient, _BaseDashaClient):
    """Standard Flower client."""

    def _gradient_step(self, parameters: NDArrays):
        """Initialize g_i with the grad (Line 2 from Alg 1 in the DASHA paper)."""
        gradients = self._calculate_gradient(parameters)
        self._gradient_estimator = gradients
        self._local_gradient_estimator = gradients
        compressed_gradient = IdentityUnbiasedCompressor().compress(
            self._gradient_estimator
        )
        return compressed_gradient

    def _compression_step(self, parameters: NDArrays):
        """Implement Lines 8 and 9 from Algorithm 1 in the DASHA paper."""
        gradients = self._calculate_gradient(parameters)
        momentum = self._get_momentum()
        assert self._local_gradient_estimator is not None
        assert self._gradient_estimator is not None
        compressed_gradient = self._compressor.compress(
            gradients
            - self._local_gradient_estimator
            - momentum * (self._gradient_estimator - self._local_gradient_estimator)
        )
        self._local_gradient_estimator = gradients
        self._gradient_estimator += decompress(compressed_gradient)
        return compressed_gradient


class MarinaClient(_GradientCompressionClient):
    """Standard Flower client."""

    def _gradient_step(self, parameters: NDArrays):
        """Implement Line 8 from Algorithm 1 in the MARINA paper if c_k = 1."""
        gradients = self._calculate_gradient(parameters)
        assert self._gradient_estimator is None
        self._local_gradient_estimator = gradients
        compressed_gradient = IdentityUnbiasedCompressor().compress(gradients)
        return compressed_gradient

    def _compression_step(self, parameters: NDArrays):
        """Implement Line 8 from Algorithm 1 in the MARINA paper if c_k = 0."""
        gradients = self._calculate_gradient(parameters)
        assert self._gradient_estimator is None
        compressed_gradient = self._compressor.compress(
            gradients - self._local_gradient_estimator
        )
        self._local_gradient_estimator = gradients
        return compressed_gradient


class _StochasticGradientCompressionClient(CompressionClient):
    _LARGE_NUMBER = 10**12

    def __init__(
        self,
        *args,
        mega_batch_size=None,
        batch_size=1,
        num_workers=4,
        evaluate_full_dataset=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._batch_size = batch_size
        assert mega_batch_size is not None
        self._mega_batch_size = mega_batch_size
        self._previous_parameters = None
        self._evaluate_full_dataset = evaluate_full_dataset
        self._batch_sampler = iter(
            torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._batch_size,
                num_workers=num_workers,
                sampler=torch.utils.data.RandomSampler(
                    self._dataset, replacement=True, num_samples=self._LARGE_NUMBER
                ),
            )
        )
        self._features, self._targets = None, None
        if self._evaluate_full_dataset:
            self._features, self._targets = _prepare_full_dataset(
                self._dataset, self._device
            )

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Send either compressed or uncompressed vector based on the config info."""
        if config[self.SEND_FULL_GRADIENT]:
            compressed_gradient = self._stochastic_gradient_step(parameters)
        else:
            compressed_gradient = self._stochastic_compression_step(parameters)
        info = {
            self.SIZE_OF_COMPRESSED_VECTORS: self._compressor.num_nonzero_components()
        }
        return (
            compressed_gradient,
            self._batch_size,
            info,
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Calculate metrics."""
        self._set_parameters(parameters)
        if not self._evaluate_full_dataset:
            features, targets = next(self._batch_sampler)
            features = features.to(self._device)
            targets = targets.to(self._device)
        else:
            features, targets = self._features, self._targets
        loss = self._function(features, targets)
        metrics = {}
        if self._evaluate_accuracy:
            accuracy = self._function.accuracy(features, targets)
            metrics[self.ACCURACY] = accuracy
        return float(loss), len(targets), metrics

    def _calculate_gradients(self, parameters, features, targets):
        """Calculate the gradient of the PyTorch model."""
        self._set_parameters(parameters)
        self._function.zero_grad()
        loss = self._function(features, targets)
        loss.backward()
        gradients = self._get_current_gradients()
        return gradients

    def _calculate_stochastic_gradient_in_current_and_previous_parameters(
        self, parameters: NDArrays
    ):
        """Calculate the stoch gradient of the PyTorch model at two points."""
        features, targets = next(self._batch_sampler)
        features = features.to(self._device)
        targets = targets.to(self._device)
        previous_gradients = self._calculate_gradients(
            self._previous_parameters, features, targets
        )
        gradients = self._calculate_gradients(parameters, features, targets)
        self._previous_parameters = parameters
        return previous_gradients, gradients

    def _calculate_mega_stochastic_gradient(self, parameters: NDArrays):
        """Calculate the stochastic gradient with large/mega batch size."""
        aggregated_gradients = 0
        for _ in range(self._mega_batch_size):
            features, targets = next(self._batch_sampler)
            features = features.to(self._device)
            targets = targets.to(self._device)
            aggregated_gradients += self._calculate_gradients(
                parameters, features, targets
            )
        aggregated_gradients /= self._mega_batch_size
        self._previous_parameters = parameters
        return aggregated_gradients

    def _stochastic_gradient_step(self, parameters: NDArrays):
        raise NotImplementedError()

    def _stochastic_compression_step(self, parameters: NDArrays):
        raise NotImplementedError()


class StochasticDashaClient(_StochasticGradientCompressionClient, _BaseDashaClient):
    """Standard Flower client."""

    def __init__(self, stochastic_momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stochastic_momentum = stochastic_momentum

    def _stochastic_gradient_step(self, parameters: NDArrays):
        """Init g_i with the stoch gradients (Line 2 from Alg 1 in the DASHA paper)."""
        gradients = self._calculate_mega_stochastic_gradient(parameters)
        self._gradient_estimator = gradients
        self._local_gradient_estimator = gradients
        compressed_gradient = IdentityUnbiasedCompressor().compress(
            self._gradient_estimator
        )
        return compressed_gradient

    def _stochastic_compression_step(self, parameters: NDArrays):
        """Implement Lines 8 and 9 from Algorithm 1 in the DASHA paper."""
        (
            previous_gradients,
            gradients,
        ) = self._calculate_stochastic_gradient_in_current_and_previous_parameters(
            parameters
        )
        next_local_gradient_estimator = gradients + (1 - self._stochastic_momentum) * (
            self._local_gradient_estimator - previous_gradients
        )
        momentum = self._get_momentum()
        assert self._local_gradient_estimator is not None
        assert self._gradient_estimator is not None
        compressed_gradient = self._compressor.compress(
            next_local_gradient_estimator
            - self._local_gradient_estimator
            - momentum * (self._gradient_estimator - self._local_gradient_estimator)
        )
        self._local_gradient_estimator = next_local_gradient_estimator
        self._gradient_estimator += decompress(compressed_gradient)
        return compressed_gradient


class StochasticMarinaClient(_StochasticGradientCompressionClient):
    """Standard Flower client."""

    def _stochastic_gradient_step(self, parameters: NDArrays):
        """Implement Line 8 in Algorithm 3 from the MARINA paper if c_k = 1."""
        gradients = self._calculate_mega_stochastic_gradient(parameters)
        assert self._gradient_estimator is None
        compressed_gradient = IdentityUnbiasedCompressor().compress(gradients)
        return compressed_gradient

    def _stochastic_compression_step(self, parameters: NDArrays):
        """Implement Line 8 in Algorithm 3 from the MARINA paper if c_k = 0."""
        (
            previous_gradients,
            gradients,
        ) = self._calculate_stochastic_gradient_in_current_and_previous_parameters(
            parameters
        )
        compressed_gradient = self._compressor.compress(gradients - previous_gradients)
        return compressed_gradient
