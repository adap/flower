"""dasha: A Flower Baseline."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import flwr as fl
from dasha.compressors import (
    IdentityUnbiasedCompressor,
    RandKCompressor,
    UnbiasedBaseCompressor,
    decompress,
)
from dasha.dataset import load_dataset, random_split
from dasha.model import ClassificationModel, define_model
from dasha.utils import _get_dataset_input_shape, reformat_config, set_seed
from flwr.client.client_app import ClientApp
from flwr.common import ArrayRecord, Context
from flwr.common.typing import NDArrays, Scalar


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
        client_state=None,
    ):  # pylint: disable=too-many-arguments
        self._function = function.to(device)
        self._function.train()
        self._compressor = (
            compressor
            if compressor is not None
            else IdentityUnbiasedCompressor()
        )
        self.client_state = client_state
        self._momentum = None
        self._evaluate_accuracy = evaluate_accuracy
        self._dataset = dataset
        self._device = device
        self._strict_load = strict_load

    def _set_parameters(self, parameters_input: NDArrays) -> None:
        """Set the parameters."""
        assert len(parameters_input) == 1
        parameters = parameters_input[0]
        self._compressor.set_dim(len(parameters))
        state_dict = {}
        shift = 0
        for k, parameter_layer in self._function.named_parameters():
            numel = parameter_layer.numel()
            end = shift + numel
            parameter = parameters[shift:end].reshape(parameter_layer.shape)
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
            [
                val.grad.cpu().numpy().flatten()
                for val in self._function.parameters()
            ]
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
        """Send either compressed or noncompressed vector based on the config.

        info.
        """
        if config[self.SEND_FULL_GRADIENT]:
            compressed_gradient = self._gradient_step(parameters)
        else:
            compressed_gradient = self._compression_step(parameters)
        payload = self._compressor.num_nonzero_components()
        info = {self.SIZE_OF_COMPRESSED_VECTORS: payload}
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
        """Initialize g_i with the grad (Line 2 from Alg 1 in the DASHA.

        paper).
        """
        gradients = self._calculate_gradient(parameters)
        self.client_state.array_records["gradient_estimator"] = ArrayRecord(
            [gradients]
        )
        self.client_state.array_records["local_gradient_estimator"] = (
            ArrayRecord([gradients])
        )
        compressed_gradient = IdentityUnbiasedCompressor().compress(gradients)
        return compressed_gradient

    def _compression_step(self, parameters: NDArrays):
        """Implement Lines 8 and 9 from Algorithm 1 in the DASHA paper."""
        gradients = self._calculate_gradient(parameters)
        momentum = self._get_momentum()
        local_gradient_estimator = self.client_state.array_records[
            "local_gradient_estimator"
        ].to_numpy_ndarrays()[0]
        gradient_estimator = self.client_state.array_records[
            "gradient_estimator"
        ].to_numpy_ndarrays()[0]
        assert local_gradient_estimator is not None
        assert gradient_estimator is not None
        compressed_gradient = self._compressor.compress(
            gradients
            - local_gradient_estimator
            - momentum * (gradient_estimator - local_gradient_estimator)
        )
        self.client_state.array_records["local_gradient_estimator"] = (
            ArrayRecord([gradients])
        )
        gradient_estimator += decompress(compressed_gradient)
        self.client_state.array_records["gradient_estimator"] = ArrayRecord(
            [gradient_estimator]
        )
        return compressed_gradient


class MarinaClient(_GradientCompressionClient):
    """Standard Flower client."""

    def _gradient_step(self, parameters: NDArrays):
        """Implement Line 8 from Algorithm 1 in the MARINA paper if c_k = 1."""
        gradients = self._calculate_gradient(parameters)
        assert (
            "gradient_estimator" not in self.client_state.array_records.keys()
        )
        self.client_state.array_records["local_gradient_estimator"] = (
            ArrayRecord([gradients])
        )
        compressed_gradient = IdentityUnbiasedCompressor().compress(gradients)
        return compressed_gradient

    def _compression_step(self, parameters: NDArrays):
        """Implement Line 8 from Algorithm 1 in the MARINA paper if c_k = 0."""
        gradients = self._calculate_gradient(parameters)
        assert (
            "gradient_estimator" not in self.client_state.array_records.keys()
        )
        compressed_gradient = self._compressor.compress(
            gradients
            - self.client_state.array_records[
                "local_gradient_estimator"
            ].to_numpy_ndarrays()[0]
        )
        self.client_state.array_records["local_gradient_estimator"] = (
            ArrayRecord([gradients])
        )
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
        self._evaluate_full_dataset = evaluate_full_dataset
        self._batch_sampler = iter(
            torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._batch_size,
                num_workers=num_workers,
                sampler=torch.utils.data.RandomSampler(
                    self._dataset,
                    replacement=True,
                    num_samples=self._LARGE_NUMBER,
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
        """Send either compressed or uncompressed vector based on the config.

        info.
        """
        if config[self.SEND_FULL_GRADIENT]:
            compressed_gradient = self._stochastic_gradient_step(parameters)
        else:
            compressed_gradient = self._stochastic_compression_step(parameters)
        payload = self._compressor.num_nonzero_components()
        info = {self.SIZE_OF_COMPRESSED_VECTORS: payload}
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

    def _calc_stoch_grad_in_curr_and_prev_params(self, parameters: NDArrays):
        """Calculate the stoch gradient of the PyTorch model at two points."""
        features, targets = next(self._batch_sampler)
        features = features.to(self._device)
        targets = targets.to(self._device)
        previous_gradients = self._calculate_gradients(
            self.client_state.array_records[
                "previous_parameters"
            ].to_numpy_ndarrays(),
            features,
            targets,
        )
        gradients = self._calculate_gradients(parameters, features, targets)
        self.client_state.array_records["previous_parameters"] = ArrayRecord(
            parameters
        )
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
        self.client_state.array_records["previous_parameters"] = ArrayRecord(
            parameters
        )
        return aggregated_gradients

    def _stochastic_gradient_step(self, parameters: NDArrays):
        raise NotImplementedError()

    def _stochastic_compression_step(self, parameters: NDArrays):
        raise NotImplementedError()


class StochasticDashaClient(
    _StochasticGradientCompressionClient, _BaseDashaClient
):
    """Standard Flower client."""

    def __init__(self, stochastic_momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stochastic_momentum = stochastic_momentum

    def _stochastic_gradient_step(self, parameters: NDArrays):
        """Init g_i with the stoch gradients (Line 2 from Alg 1 in the DASHA.

        paper).
        """
        gradients = self._calculate_mega_stochastic_gradient(parameters)
        self.client_state.array_records["gradient_estimator"] = ArrayRecord(
            [gradients]
        )
        self.client_state.array_records["local_gradient_estimator"] = (
            ArrayRecord([gradients])
        )
        compressed_gradient = IdentityUnbiasedCompressor().compress(
            self.client_state.array_records[
                "gradient_estimator"
            ].to_numpy_ndarrays()[0]
        )
        return compressed_gradient

    def _stochastic_compression_step(self, parameters: NDArrays):
        """Implement Lines 8 and 9 from Algorithm 1 in the DASHA paper."""
        (
            previous_gradients,
            gradients,
        ) = self._calc_stoch_grad_in_curr_and_prev_params(parameters)
        local_gradient_estimator = self.client_state.array_records[
            "local_gradient_estimator"
        ].to_numpy_ndarrays()[0]
        gradient_estimator = self.client_state.array_records[
            "gradient_estimator"
        ].to_numpy_ndarrays()[0]
        next_local_gradient_estimator = gradients + (
            1 - self._stochastic_momentum
        ) * (local_gradient_estimator - previous_gradients)
        momentum = self._get_momentum()
        assert local_gradient_estimator is not None
        assert gradient_estimator is not None
        compressed_gradient = self._compressor.compress(
            next_local_gradient_estimator
            - local_gradient_estimator
            - momentum * (gradient_estimator - local_gradient_estimator)
        )
        self.client_state.array_records["local_gradient_estimator"] = (
            ArrayRecord([next_local_gradient_estimator])
        )
        gradient_estimator += decompress(compressed_gradient)
        self.client_state.array_records["gradient_estimator"] = ArrayRecord(
            [gradient_estimator]
        )
        return compressed_gradient


class StochasticMarinaClient(_StochasticGradientCompressionClient):
    """Standard Flower client."""

    def _stochastic_gradient_step(self, parameters: NDArrays):
        """Implement Line 8 in Algorithm 3 from the MARINA paper if c_k = 1."""
        gradients = self._calculate_mega_stochastic_gradient(parameters)
        assert (
            "gradient_estimator" not in self.client_state.array_records.keys()
        )
        compressed_gradient = IdentityUnbiasedCompressor().compress(gradients)
        return compressed_gradient

    def _stochastic_compression_step(self, parameters: NDArrays):
        """Implement Line 8 in Algorithm 3 from the MARINA paper if c_k = 0."""
        (
            previous_gradients,
            gradients,
        ) = self._calc_stoch_grad_in_curr_and_prev_params(parameters)
        compressed_gradient = self._compressor.compress(
            gradients - previous_gradients
        )
        return compressed_gradient


def define_client_obj(
    method_cfg: dict, function, dataset, compressor, client_state
):
    """Generate the client object for each method."""
    method_name = method_cfg["name"]
    if method_name in ("dasha", "marina"):
        client_obj = DashaClient if method_name == "dasha" else MarinaClient
        return client_obj(
            function=function,
            dataset=dataset,
            compressor=compressor,
            client_state=client_state,
            device=method_cfg["device"],
            evaluate_accuracy=method_cfg["evaluate-accuracy"],
            send_gradient=method_cfg["send-gradient"],
            strict_load=method_cfg["strict-load"],
        )
    if method_name == "stochastic_dasha":
        return StochasticDashaClient(
            function=function,
            dataset=dataset,
            compressor=compressor,
            client_state=client_state,
            device=method_cfg["device"],
            evaluate_accuracy=method_cfg["evaluate-accuracy"],
            strict_load=method_cfg["strict-load"],
            stochastic_momentum=method_cfg["stochastic-momentum"],
            batch_size=method_cfg["batch-size"],
            mega_batch_size=method_cfg["mega-batch-size"],
        )
    return StochasticMarinaClient(
        function=function,
        dataset=dataset,
        compressor=compressor,
        client_state=client_state,
        device=method_cfg["device"],
        evaluate_accuracy=method_cfg["evaluate-accuracy"],
        strict_load=method_cfg["strict-load"],
        batch_size=method_cfg["batch-size"],
        mega_batch_size=method_cfg["mega-batch-size"],
    )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    cfg = reformat_config(context.run_config)
    seed = set_seed(seed=42)
    dataset = load_dataset(cfg)
    datasets = random_split(dataset, num_partitions)
    local_dataset = datasets[partition_id]
    model = define_model(
        cfg["model"], input_shape=_get_dataset_input_shape(dataset)
    )
    compressor = RandKCompressor(
        number_of_coordinates=cfg["compressor"]["number-of-coordinates"],
        seed=seed,
    )
    # Return Client instance
    client_state = context.state
    client_instance = define_client_obj(
        cfg["method"],
        function=model,
        dataset=local_dataset,
        compressor=compressor,
        client_state=client_state,
    )
    # Return Client instance
    return client_instance.to_client()


# Flower ClientApp
app = ClientApp(client_fn)
