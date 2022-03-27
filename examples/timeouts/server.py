import flwr as fl

# Start Flower server with a request_timeout of 3 seconds
# This example has two fast clients which return withing 1 second
# It also has one slow client which returns within 5 seconds
# The first time the slow client is sampled the server will time it out
# and we will have 2 our of 3 requests beeing successful.
# The slow client will than crash as it will receive a DEADLINE_EXCEEDED
# response from the server which will terminate the connection.
fl.server.start_server(
    strategy=fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
    ),
    server_address="[::]:8080",
    config={"num_rounds": 3},
    request_timeout=3,
)
