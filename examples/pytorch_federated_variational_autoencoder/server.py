import flwr as fl


def main():
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 3},
    )


if __name__ == "__main__":
    main()
