import flwr as fl
import numpy as np


class NewtonRaphsonStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        inplace=True,
        damping_factor=0.8,
    ):
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
            inplace=inplace,
        )
        self.damping_factor = damping_factor

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        n_all_samples = sum([res[1] for res in weights_results])

        total_hessians = None
        total_gradient_one_d = None

        last_gradients = None

        for ndarrays, num_samples in weights_results:

            sample_coefficient = num_samples / n_all_samples

            gradients = ndarrays[:-1]
            hessian = ndarrays[-1]

            last_gradients = gradients

            if total_hessians is None or total_gradient_one_d is None:
                total_hessians = hessian * sample_coefficient
                total_gradient_one_d = (
                    np.concatenate([grad.reshape(-1) for grad in gradients])
                    * sample_coefficient
                )
            else:
                total_hessians += hessian * sample_coefficient
                total_gradient_one_d += (
                    np.concatenate([grad.reshape(-1) for grad in gradients])
                    * sample_coefficient
                )

        assert total_hessians is not None and total_gradient_one_d is not None

        parameters_update = -self.damping_factor * np.linalg.solve(
            total_hessians, total_gradient_one_d
        )
        # Why use the last gradient?
        parameters_aggregated = fl.common.ndarrays_to_parameters(
            self._unflatten_array(parameters_update, last_gradients)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            print("No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def _unflatten_array(self, array_one_d, list_of_array):
        assert (
            len(array_one_d.shape) == 1
        )  # The array to unflatten have to be a 1 dimensional array

        result = []
        current_index = 0

        for array in list_of_array:
            num_params = len(array.ravel())
            result.append(
                np.array(
                    array_one_d[current_index : current_index + num_params].reshape(
                        array.shape
                    )
                )
            )
            current_index += num_params

        return result
