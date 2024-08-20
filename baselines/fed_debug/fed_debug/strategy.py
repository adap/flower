"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

import gc
from flwr.common.logger import log
from logging import INFO

import flwr as fl


from fed_debug.models import initialize_model
from fed_debug import utils


class FedAvgSave(fl.server.strategy.FedAvg):
    """SaveModelStrategy."""

    def __init__(self, cfg, cache, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.cache = cache

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate clients updates."""
        round_dict = {}
        round_dict["client2ws"] = {
            fit_res.metrics["cid"]: self.get_state_dict_from_parameters(
                fit_res.parameters
            )
            for _, fit_res in results
        }

        client_ids = round_dict["client2ws"].keys()

        log(INFO, f"participating clients: {client_ids}")

        # client2num_examples save in round_dict from results
        round_dict["client2num_examples"] = {
            fit_res.metrics["cid"]: fit_res.num_examples for _, fit_res in results
        }
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        round_key = f"{self.cfg.exp_key}-round:{server_round}"

        if aggregated_parameters is not None:
            round_dict["gm_ws"] = self.get_state_dict_from_parameters(
                aggregated_parameters
            )
            self.cache[round_key] = round_dict
            round_dict.clear()
        del results
        gc.collect()
        return aggregated_parameters, aggregated_metrics

    def get_state_dict_from_parameters(self, parameters):
        """Convert parameters to state_dict."""
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        temp_net = initialize_model(self.cfg.model.name, self.cfg.dataset)["model"]
        utils.set_parameters(temp_net, ndarr)
        return temp_net.state_dict()
