import argparse
import os

import tensorflow as tf

import flwr as fl

import common

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, test = tf.keras.datasets.mnist.load_data()
    test_data, test_labels = test

    # preprocessing
    test_data, test_labels = common.preprocess(test_data, test_labels)

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights):
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(test_data, test_labels)
        return loss, {"accuracy": accuracy}

    return evaluate


def main(args) -> None:
    model = common.create_cnn_model()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile("sgd", loss=loss, metrics=["accuracy"])
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        min_available_clients=args.num_clients,
        evaluate_fn=get_evaluate_fn(model),
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config={"num_rounds": args.num_rounds},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-clients", default=2, type=int)
    parser.add_argument("--num-rounds", default=1, type=int)
    parser.add_argument("--fraction-fit", default=1.0, type=float)
    args = parser.parse_args()
    main(args)
