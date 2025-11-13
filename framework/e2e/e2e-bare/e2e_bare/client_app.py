from datetime import datetime

import numpy as np

from flwr.app import ConfigRecord, Context, RecordDict
from flwr.client import NumPyClient, start_client
from flwr.clientapp import ClientApp

SUBSET_SIZE = 1000
STATE_VAR = "timestamp"


model_params = np.array([1])
objective = 5


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, state: RecordDict):
        self.state = state

    def get_parameters(self, config):
        return model_params

    def _record_timestamp_to_state(self):
        """Record timestamp to client's state."""
        t_stamp = datetime.now().timestamp()
        value = str(t_stamp)
        if STATE_VAR in self.state.config_records.keys():
            value = self.state.config_records[STATE_VAR][STATE_VAR]  # type: ignore
            value += f",{t_stamp}"

        self.state.config_records[STATE_VAR] = ConfigRecord({STATE_VAR: value})

    def _retrieve_timestamp_from_state(self):
        return self.state.config_records[STATE_VAR][STATE_VAR]

    def fit(self, parameters, config):
        model_params = parameters
        model_params = [param * (objective / np.mean(param)) for param in model_params]
        self._record_timestamp_to_state()
        return model_params, 1, {STATE_VAR: self._retrieve_timestamp_from_state()}

    def evaluate(self, parameters, config):
        model_params = parameters
        loss = min(np.abs(1 - np.mean(model_params) / objective), 1)
        accuracy = 1 - loss
        self._record_timestamp_to_state()
        return (
            loss,
            1,
            {"accuracy": accuracy, STATE_VAR: self._retrieve_timestamp_from_state()},
        )


def client_fn(context: Context):
    return FlowerClient(context.state).to_client()


app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    # Start Flower client
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(state=RecordDict()).to_client(),
    )
