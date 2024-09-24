"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

# import all the necessary libraries
import os
import re
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import mmcv
import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy


class FedVSSL(fl.server.strategy.FedAvg):
    """Define the custom strategy for FedVSSL."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_rounds: int = 1,
        mix_coeff: float = 0.2,
        swbeta: int = 0,
        base_work_dir: str = "ucf_FedVSSL",
        fedavg: bool = False,
        **kwargs,
    ):
        assert isinstance(swbeta, int) and swbeta in [
            0,
            1,
        ], "the value must be an integer and either 0 or 1 "
        self.num_rounds = (num_rounds,)
        self.mix_coeff = (
            mix_coeff  # coefficient for mixing the loss and fedavg aggregation methods
        )
        self.swbeta = swbeta  # 0: SWA off; 1:SWA on
        self.base_work_dir = base_work_dir
        self.fedavg = fedavg
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Divide the weights based on the backbone and classification head
        ###################################################################

        # Aggregate all the weights and the number of examples
        weight_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        # aggregate the weights of the backbone
        weights_avg = aggregate(weight_results)  # Fedavg

        if self.fedavg:
            weights_avg = ndarrays_to_parameters(weights_avg)
            glb_dir = self.base_work_dir
            mmcv.mkdir_or_exist(os.path.abspath(glb_dir))
        else:
            # Aggregate all the weights and the loss
            loss_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics["loss"])
                for client, fit_res in results
            ]

            # aggregate the weights of the backbone
            weights_loss = aggregate(loss_results)  # loss-based

            weights: NDArrays = [v * self.mix_coeff for v in weights_avg] + [
                (1 - self.mix_coeff) * v for v in weights_loss
            ]  # Equation 3 in FedVSSL paper

            # # aggregate the weights of the backbone
            # weights = aggregate(weight_results) # loss-based

            # create a directory to save the global checkpoints
            glb_dir = self.base_work_dir
            mmcv.mkdir_or_exist(os.path.abspath(glb_dir))

            # load the previous weights if there are any

            if server_round > 1 and self.swbeta == 1:
                chk_name_list = [
                    fn for fn in os.listdir(glb_dir) if fn.endswith(".npz")
                ]
                chk_epoch_list = [
                    int(re.findall(r"\d+", fn)[0])
                    for fn in chk_name_list
                    if fn.startswith("round")
                ]
                if chk_epoch_list:
                    chk_epoch_list.sort()
                    print(chk_epoch_list)
                    # select the most recent epoch
                    checkpoint = os.path.join(
                        glb_dir, f"round-{chk_epoch_list[-1]}-weights.array.npz"
                    )
                    # load the previous model weights
                    params = np.load(checkpoint, allow_pickle=True)
                    params = params["arr_0"].item()
                    print("The weights has been loaded")
                    params = parameters_to_ndarrays(params)  # return a list
                    weights_avg = [
                        np.asarray((0.5 * B + 0.5 * A)) for A, B in zip(weights, params)
                    ]  # perform SWA
                    weights_avg = ndarrays_to_parameters(weights_avg)
            else:
                print("The results are saved without performing SWA")
                weights_avg = ndarrays_to_parameters(weights)

        if weights_avg is not None:
            # save weights
            print(
                f"round-{server_round}-weights...",
            )
            np.savez(
                os.path.join(glb_dir, f"round-{server_round}-weights.array"),
                weights_avg,
            )

        return weights_avg, {}


def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
