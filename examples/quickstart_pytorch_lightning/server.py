import flwr as fl


def main() -> None:
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 10},
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
