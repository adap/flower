"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from collections import Counter

import flwr as fl

from feddebug import utils
from feddebug.differential_testing import differential_testing_fl_clients


class FedAvgWithFedDebug(fl.server.strategy.FedAvg):
    """FedAvg with Differential Testing."""

    def __init__(
        self,
        num_bugs,
        num_inputs,
        input_shape,
        na_t,
        device,
        fast,
        callback_create_model_fn,
        callback_fed_debug_evaluate_fn,
        *args,
        **kwargs,
    ):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.num_bugs = num_bugs
        self.num_inputs = num_inputs
        self.na_t = na_t
        self.device = device
        self.fast = fast
        self.create_model_fn = callback_create_model_fn
        self.callback_fed_debug_evaluate_fn = callback_fed_debug_evaluate_fn

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate clients updates."""
        potential_mal_clients = self._run_differential_testing_helper(results)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        aggregated_metrics["potential_malicious_clients"] = potential_mal_clients
        self.callback_fed_debug_evaluate_fn(server_round, potential_mal_clients)
        return aggregated_parameters, aggregated_metrics

    def _get_model_from_parameters(self, parameters):
        """Convert parameters to state_dict."""
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        temp_net = self.create_model_fn()
        utils.set_parameters(temp_net, ndarr)
        return temp_net

    def _run_differential_testing_helper(self, results):
        client2model = {
            fit_res.metrics["cid"]: self._get_model_from_parameters(fit_res.parameters)
            for _, fit_res in results
        }
        predicted_faulty_clients_on_each_input = differential_testing_fl_clients(
            client2model,
            self.num_bugs,
            self.num_inputs,
            self.input_shape,
            self.na_t,
            self.fast,
            self.device,
        )
        mal_clients_dict = Counter(
            [
                f"{e}"
                for temp_set in predicted_faulty_clients_on_each_input
                for e in temp_set
            ]
        )
        return dict(mal_clients_dict)
