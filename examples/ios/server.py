import flwr


def main() -> None:
# Start Flower server
    strategy = flwr.server.strategy.FedAvg(
        min_fit_clients=1, 
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    flwr.server.start_server(
        server_address="[::]:8080", 
        config=flwr.server.ServerConfig(num_rounds=1), 
        strategy=strategy,
    )

if __name__ == "__main__":
    main()