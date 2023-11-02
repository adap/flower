from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
import json

from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log


class XGbBagging(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using bagging."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate all the client trees
        global_model = None
        for _, fit_res in results:
            update = fit_res.parameters.tensors
            for item in update:
                global_model = aggregate(global_model, json.loads(bytearray(item)))

        weights_avg = json.dumps(global_model)

        return (
            Parameters(
                tensor_type="", tensors=[bytes(weights_avg, "utf-8")]
            ),
            {},
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics using average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return 0, metrics_aggregated


def aggregate(bst_prev, bst_curr):
    if not bst_prev:
        return bst_curr
    else:
        # get the tree numbers
        tree_num_prev, paral_tree_num_prev = _get_tree_nums(bst_prev)
        tree_num_curr, paral_tree_num_curr = _get_tree_nums(bst_curr)

        bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ] = str(tree_num_prev + paral_tree_num_curr)
        iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
            "iteration_indptr"
        ]
        bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
            iteration_indptr[-1] + 1 * paral_tree_num_curr
        )

        # aggregate new trees
        trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
        for tree_count in range(paral_tree_num_curr):
            trees_curr[tree_count]["id"] = tree_num_prev + tree_count
            bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
                trees_curr[tree_count]
            )
            bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
        return bst_prev


def _get_tree_nums(xgb_model):
    # get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num
