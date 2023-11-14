import json
import os
from collections import OrderedDict

import torch
from centralized import get_model

import flwr as fl
from flwr.server import History


def trainconfig(server_round):
    """Return training configuration dict for each round."""
    config = {"server_round": server_round}  # The current round of federated learning
    return config


class ClientManager(fl.server.SimpleClientManager):
    def sample(
        self,
        num_clients,
        server_round,
        min_num_clients,
        criterion,
    ):
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)

        # First round and odd rounds
        if server_round >= 1 and num_clients <= 20:
            available_cids_sorted = sorted(available_cids, key=int)
            section = []
            for j in range(4):
                index = ((server_round - 1) * 4 + j) % len(available_cids_sorted)
                section.append(available_cids_sorted[index])
            available_cids = section.copy()
            print("Available cids: ", available_cids)
            return [self.clients[cid] for cid in available_cids]

        if num_clients > 20:
            print(
                "Support for distributed training of UNet in specified hardware for larger clients is unavailable"
            )


class SaveModelAndMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        args,
        *,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
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
        client_manager=None,
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
        )
        self.client_manager = client_manager
        self.history = History()
        self.args = args

    def aggregate_fit(
        self,
        server_round,
        results,  # FitRes is like EvaluateRes and has a metrics key
        failures,
    ):
        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            model = get_model()
            params_dict = zip(model.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            output_dir = self.args.model_path
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{output_dir}/model_round_{server_round}.pth",
            )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        aggregated_loss = 1.0

        if server_round > 5:
            cid_list = [r.metrics["cid"] for _, r in results]
            precision_list = [r.metrics["precision"] for _, r in results]
            recall_list = [r.metrics["recall"] for _, r in results]
            print(cid_list, precision_list, recall_list, server_round)

            lowest_precision = float("inf")
            lowest_precision_cid = None
            lowest_precision_count = 0
            for i, precision in enumerate(precision_list):
                if float(precision) < float(lowest_precision):
                    lowest_precision = precision
                    lowest_precision_cid = cid_list[i]
                    lowest_precision_count = 1
                elif precision == lowest_precision:
                    lowest_precision_count += 1

            if lowest_precision_count > 1:
                lowest_precision_cid = None

            lowest_precision_cid = str(lowest_precision_cid)

            client_to_disconnect = None
            for client, evaluate_res in results:
                if evaluate_res.metrics.get("cid") == lowest_precision_cid:
                    client_to_disconnect = client

            if lowest_precision_cid == None:
                loss_aggregated, metrics_aggregated = aggregated_loss, {
                    "server_round": server_round
                }
            else:
                print(
                    "client_to_disconnect:", lowest_precision_cid, client_to_disconnect
                )
                print("====done with agg evaluate======")

                data = {
                    "cid_list": cid_list,
                    "precision_list": precision_list,
                    "recall_list": recall_list,
                    "lowest_precision_cid": lowest_precision_cid,
                    "server_round": server_round,
                    "client_to_disconnect": client_to_disconnect,
                    "warning_client": 0,
                }
                print(data)
                # Serialize data into file:
                os.makedirs(self.args.log_path, exist_ok=True)
                json.dump(data, open(f"{self.args.log_path}/logs.json", "a"))

                loss_aggregated, metrics_aggregated = aggregated_loss, data
        else:
            loss_aggregated, metrics_aggregated = aggregated_loss, {
                "client_to_disconnect": None,
                "server_round": server_round,
            }

        if self.client_manager:
            print("For Personalization and Threshold Filtering Strategy")
            print("History=====")
            warning_clients = [
                metric
                for _, metric in self.history.metrics_distributed.get(
                    "warning_client", []
                )
            ]
            if len(warning_clients) > 1:
                if int(metrics_aggregated["lowest_precision_cid"]) in warning_clients:
                    print("Disconnecting after 2 attempts ...")
                    if (
                        int(metrics_aggregated["server_round"]) >= 1
                        and metrics_aggregated["client_to_disconnect"] is not None
                    ):
                        client_to_disconnect = metrics_aggregated[
                            "client_to_disconnect"
                        ]
                        lowest_precision_cid = metrics_aggregated[
                            "lowest_precision_cid"
                        ]
                        print(
                            "client_to_disconnect:",
                            lowest_precision_cid,
                            client_to_disconnect,
                        )
                        self.client_manager.unregister(client_to_disconnect)
                        print("=====disconnected=====")
                        all_clients = self.client_manager.all()
                        clients = list(all_clients.keys())
                        print(
                            f"Clients still connected after Server round {metrics_aggregated['server_round']}:{clients}"
                        )
            if (
                int(metrics_aggregated["server_round"]) >= 1
                and metrics_aggregated["client_to_disconnect"] is None
            ):
                print("Nothing to disconnect")

        self.history.add_metrics_distributed(server_round, metrics_aggregated)

        return loss_aggregated, metrics_aggregated
