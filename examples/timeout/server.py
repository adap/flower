import flwr as fl

# Start Flower server with a request_timeout of 3 seconds.
# This example assumes two fast clients which return within 1 second.
# It also assumes one slow client which returns within 5 seconds.
# The first time the slow client is sampled, the server will time it out
# and we will have 2 of our 3 requests being successful.
# The slow client will then crash as it will receive a DEADLINE_EXCEEDED
# response from the server which will terminate the connection.
# Subsequent rounds will only see two available clients.
fl.server.start_server(
    strategy=fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
    ),
    server_address="[::]:8080",
    config={"num_rounds": 3, "round_timeout": 30.0},
)
