# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FedRDF: A Robust and Dynamic Aggregation Function Against Poisoning Attacks.

Reference:
E. Mármol Campos, A. González-Vidal, J. L. Hernández-Ramos, and A. Skarmeta,
"FedRDF: A Robust and Dynamic Aggregation Function Against Poisoning Attacks in
Federated Learning," IEEE Transactions on Emerging Topics in Computing, vol. 13,
no. 1, pp. 48–67, 2025. DOI: 10.1109/TETC.2024.3474484

Paper: https://ieeexplore.ieee.org/abstract/document/10713851
"""

from functools import reduce
from logging import INFO, WARNING
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import ks_2samp

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from .fedavg import FedAvg


class FedRDF(FedAvg):
    """Robust and Dynamic Aggregation Function (FedRDF) strategy.

    FedRDF adaptively switches between standard FedAvg and FFT-based robust
    aggregation depending on the detected skewness in client updates. High
    skewness indicates potential poisoning attacks, triggering the robust
    FFT aggregation mechanism.

    The strategy analyzes client weight distributions using statistical tests
    (Kolmogorov-Smirnov) to compute skewness. When skewness exceeds a threshold,
    it applies Fourier Transform-based aggregation to mitigate the impact of
    poisoned updates.

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    threshold : float, optional
        Skewness threshold for switching between FedAvg and FFT aggregation.
        - If threshold <= 0: Always use FFT-based robust aggregation
        - If threshold > 0: Use FFT only when detected skewness > threshold
        Defaults to 0.0 (always use FFT).

    Reference
    ---------
    E. Mármol Campos et al., "FedRDF: A Robust and Dynamic Aggregation Function
    Against Poisoning Attacks in Federated Learning," IEEE TETC, 2025.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        threshold: float = 0.0,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.threshold = threshold

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        return f"FedRDF(accept_failures={self.accept_failures}, threshold={self.threshold})"

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using FedRDF adaptive aggregation.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful client updates from the current round.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred during client training.

        Returns
        -------
        parameters : Optional[Parameters]
            Aggregated model parameters, or None if aggregation failed.
        metrics : Dict[str, Scalar]
            Aggregated metrics from clients.
        """
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results to (weights, num_examples) tuples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Perform FedRDF aggregation
        parameters_aggregated = ndarrays_to_parameters(
            self.aggregate_fedrdf(weights_results)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def ks_proportion(self, sample: np.ndarray) -> float:
        """Estimate distribution divergence using Kolmogorov-Smirnov test.

        Repeatedly samples random subsets from the data and tests if they come
        from the same distribution as the remaining data. Returns the proportion
        of tests that reject the null hypothesis (indicating divergence).

        Parameters
        ----------
        sample : np.ndarray
            Array of values to test for distribution divergence.

        Returns
        -------
        float
            Proportion of K-S tests that detected significant divergence (0-1).
        """
        total = []
        significance_level = 0.05

        for _ in range(1000):
            # Randomly select 30% of the sample
            n_random = int(len(sample) * 0.3)
            random_indices = np.random.choice(
                len(sample), size=n_random, replace=False
            )
            random_points = sample[random_indices]
            generated_points = np.delete(sample, random_indices)

            # Perform Kolmogorov-Smirnov test
            _, p_value = ks_2samp(generated_points, random_points)

            # Record if distributions are significantly different
            total.append(1 if p_value < significance_level else 0)

        return np.mean(total)

    def skewness(self, arrays: List[np.ndarray]) -> float:
        """Measure the skewness proportion across client weight arrays.

        Analyzes the distribution of weight values at each position across clients.
        High skewness indicates that some weights have significantly different
        distributions, which may signal poisoning attacks.

        Parameters
        ----------
        arrays : List[np.ndarray]
            List of weight arrays from different clients (same shape).

        Returns
        -------
        float
            Average skewness proportion across all weight positions (0-1).
        """
        # Stack arrays into matrix: (n_clients, n_weights)
        stacked = np.stack([arr.flatten() for arr in arrays], axis=0)

        # Standardize each weight position across clients
        means = stacked.mean(axis=0)
        stds = stacked.std(axis=0)
        # Avoid division by zero for constant weights
        stds = np.where(stds == 0, 1.0, stds)
        standardized = (stacked - means) / stds

        # Compute K-S proportion for each weight position
        proportions = []
        for i in range(standardized.shape[1]):
            prop = self.ks_proportion(standardized[:, i])
            proportions.append(prop)

        return np.mean(proportions)

    def fourier_aggregate(self, arrays: List[np.ndarray]) -> np.ndarray:
        """Aggregate weights using FFT-based robust aggregation.

        For each weight position, collects values across all clients, computes
        the FFT, and selects the value corresponding to the dominant frequency
        component. This approach is robust to outliers in the frequency domain.

        Parameters
        ----------
        arrays : List[np.ndarray]
            List of weight arrays from different clients (same shape).

        Returns
        -------
        np.ndarray
            Aggregated weight array with the same shape as input arrays.
        """
        result = np.zeros_like(arrays[0])
        flat_size = arrays[0].size

        for i in range(flat_size):
            # Collect values at position i from all clients
            values = np.array([arr.flat[i] for arr in arrays])

            # Compute FFT to analyze frequency components
            fft_values = np.fft.fft(values)

            # Find the dominant frequency component
            magnitudes = np.abs(fft_values)
            dominant_idx = np.argmax(magnitudes)

            # Use the original value corresponding to the dominant frequency
            result.flat[i] = values[dominant_idx]

        return result

    def aggregate_fedrdf(
        self, results: List[Tuple[NDArrays, int]]
    ) -> NDArrays:
        """Perform FedRDF aggregation with adaptive switching.

        Computes skewness in client updates and adaptively chooses between:
        - Standard weighted averaging (FedAvg) for normal scenarios
        - FFT-based robust aggregation for high-skewness scenarios (attacks)

        Parameters
        ----------
        results : List[Tuple[NDArrays, int]]
            List of (weights, num_examples) tuples from clients.

        Returns
        -------
        NDArrays
            Aggregated model weights (list of numpy arrays, one per layer).
        """
        num_examples_total = sum(num_examples for _, num_examples in results)
        weights_list = [weights for weights, _ in results]

        # Always use FFT if threshold <= 0
        if self.threshold <= 0:
            log(
                INFO,
                f"FedRDF round: Using FFT-based robust aggregation (threshold={self.threshold})",
            )
            aggregated_weights: NDArrays = [
                self.fourier_aggregate(layers) for layers in zip(*weights_list)
            ]
            return aggregated_weights

        # Compute skewness for each layer
        skewness_scores = [
            self.skewness(layers) for layers in zip(*weights_list)
        ]
        avg_skewness = np.mean(skewness_scores)

        log(
            INFO,
            f"FedRDF round: Detected skewness={avg_skewness:.4f}, threshold={self.threshold}",
        )

        # Adaptive decision based on skewness
        if avg_skewness < self.threshold:
            # Low skewness: Use standard FedAvg (weighted by num_examples)
            log(
                INFO,
                "FedRDF round: Using standard FedAvg (low skewness, benign scenario)",
            )
            weighted_weights = [
                [layer * num_examples for layer in weights]
                for weights, num_examples in results
            ]
            aggregated_weights = [
                reduce(np.add, layer_updates) / num_examples_total
                for layer_updates in zip(*weighted_weights)
            ]
        else:
            # High skewness: Use FFT-based robust aggregation
            log(
                INFO,
                "FedRDF round: Using FFT-based robust aggregation (high skewness detected, potential attack)",
            )
            aggregated_weights = [
                self.fourier_aggregate(layers) for layers in zip(*weights_list)
            ]

        return aggregated_weights
