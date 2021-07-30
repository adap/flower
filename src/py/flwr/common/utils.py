from ..server.strategy.aggregate import weighted_loss_avg
import tensorflow as tf
import os, os.path


def tensorboard_writer(argument):
    def decorator(func):
        def wrapper(*args, **kwargs):
            rnd, results = args[1], args[2]
            writer_distributed = tf.summary.create_file_writer(
                f"{argument}/distributed"
            )

            if rnd != -1:
                step = rnd
            else:
                step = len(
                    [
                        name
                        for name in os.listdir(f"{argument}/distributed")
                        if os.path.isfile(os.path.join(f"{argument}/distributed", name))
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
                        f"accuracy_client_{client_idx+1}",
                        evaluate_res.accuracy,
                        step=step,
                    )
                writer_distributed.flush()

            writer_federated = tf.summary.create_file_writer(f"{argument}/federated")
            with writer_federated.as_default():
                loss_aggregated = weighted_loss_avg(
                    [
                        (
                            evaluate_res.num_examples,
                            evaluate_res.loss,
                            evaluate_res.accuracy,
                        )
                        for _, evaluate_res in results
                    ]
                )
                tf.summary.scalar(f"loss_aggregated", loss_aggregated, step=step)
                writer_federated.flush()

            return func(*args, **kwargs)

        return wrapper

    return decorator
