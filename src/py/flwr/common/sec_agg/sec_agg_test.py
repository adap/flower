import flwr as fl
from flwr.common import GRPC_MAX_MESSAGE_LENGTH

from flwr.server.strategy.sec_agg_fedavg import SecAggFedAvg

# Testing


def test_start_server(vector_dimension=100000, dropout_value=0):
    fl.server.start_server("localhost:8080", config={
                           "num_rounds": 1, "sec_agg": 1},
                           strategy=SecAggFedAvg(sec_agg_param_dict={"test": 1,
                                                                     "test_vector_dimension": vector_dimension,
                                                                     "test_dropout_value": dropout_value}))


def test_start_client(server_address: str,
                      client,
                      grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,):
    fl.client.start_numpy_client(server_address, client, grpc_max_message_length)
