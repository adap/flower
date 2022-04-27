import flwr as fl


def main():
    fl.server.start_server(
        config={"num_rounds": 3},
    )


if __name__ == "__main__":
    main()
