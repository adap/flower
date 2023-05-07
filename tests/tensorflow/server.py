import sys

import flwr as fl


# Start Flower server
hist = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
)

prev_loss = 1_000_000
for _, loss in hist.losses_distributed:
    if loss > prev_loss:
        with open("result", "w") as file:
            file.write("FAIL")
        sys.exit("Loss did not decrease.")
    else:
        prev_loss = loss


with open("result", "w") as file:
    file.write("SUCCESS")

sys.exit()
