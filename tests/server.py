import flwr as fl
from flwr.server.strategy.fedavg import FedAvg

from flwr.server.strategy.sec_agg_fedavg import SecAggFedAvg
from flwr.common.sec_agg import sec_agg_test

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    #sec_agg_test.test_start_server(vector_dimension=100000, dropout_value=0)

    fl.server.start_server("localhost:8080", config={
        "num_rounds": 1, "sec_agg": 1}, strategy=SecAggFedAvg())

'''
# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.server.start_server("localhost:8080", config={
                           "num_rounds": 1}, strategy=FedAvg())'''
