import flwr as fl
import tensorflow as tf
from typing import Optional
from typing import List, Dict, Optional, Tuple
from flwr.common import Scalar
import os, os.path


def weighted_loss_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _, _ in results]
    )
    weighted_losses = [num_examples * loss for num_examples, loss, _ in results]
    return sum(weighted_losses) / num_total_evaluation_examples


class TensorBoardStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self, rnd: int, results, failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        loss_aggregated = super().aggregate_evaluate(rnd, results, failures)[0]
        writer_distributed = tf.summary.create_file_writer("mylogs/distributed")

        if rnd != -1:
            step = rnd
        else:
            step = len(
                [
                    name
                    for name in os.listdir("mylogs/distributed")
                    if os.path.isfile(os.path.join("mylogs/distributed", name))
                ]
            )

        with writer_distributed.as_default():
            for client_idx, (_, evaluate_res) in enumerate(results):
                tf.summary.scalar(
                    f"num_examples_client_{client_idx+1}",
                    evaluate_res.num_examples,
                    step=step,
                )
                tf.summary.scalar(
                    f"loss_client_{client_idx+1}", evaluate_res.loss, step=step
                )
                tf.summary.scalar(
                    f"accuracy_client_{client_idx+1}", evaluate_res.accuracy, step=step
                )
            writer_distributed.flush()

        writer_federated = tf.summary.create_file_writer("mylogs/federated")
        with writer_federated.as_default():

            tf.summary.scalar(f"loss_aggregated", loss_aggregated, step=step)
            writer_federated.flush()

        return loss_aggregated, {}
