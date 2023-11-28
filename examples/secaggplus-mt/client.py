import time

import numpy as np

import flwr as fl
from flwr.common import Status, FitIns, FitRes, Code
from flwr.common.parameter import ndarrays_to_parameters
from flwr.client.secure_aggregation import secaggplus_middleware


# Define Flower client with the SecAgg+ protocol
class FlowerClient(fl.client.Client):
    def fit(self, fit_ins: FitIns) -> FitRes:
        ret_vec = [np.ones(3)]
        ret = FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(ret_vec),
            num_examples=1,
            metrics={},
        )
        # Force a significant delay for testing purposes
        if fit_ins.config["drop"]:
            print(f"Client dropped for testing purposes.")
            time.sleep(4)
            return ret
        print(f"Client uploading {ret_vec[0]}...")
        return ret


def client_fn(cid: str):
    """."""
    return FlowerClient().to_client()


# To run this: `flower-client --callable client:flower`
flower = fl.flower.Flower(
    client_fn=client_fn,
    middleware=[secaggplus_middleware],
)


if __name__ == "__main__":
    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:9092",
        client=FlowerClient(),
        transport="grpc-rere",
    )
