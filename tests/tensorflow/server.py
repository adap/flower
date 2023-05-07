import sys

import flwr as fl


# Start Flower server
hist = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
)

improvement = hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]
if improvement < 0.98:
    with open("result", "w") as file:
        file.write("FAIL")
    sys.exit("Loss did not decrease.")
else:
    with open("result", "w") as file:
        file.write("SUCCESS")
    sys.exit()
