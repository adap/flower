import click
from flwr.server import start_server
from flwr.server.strategy.fast_and_slow import FastAndSlow
from flwr.server.strategy.fault_tolerant_fedavg import FaultTolerantFedAvg
from flwr.server.strategy.fedadagrad import FedAdagrad
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.strategy.fedfs_v0 import FedFSv0
from flwr.server.strategy.fedfs_v1 import FedFSv1
from flwr.server.strategy.fedopt import FedOpt
from flwr.server.strategy.qfedavg import QFedAvg
from flwr.common import GRPC_MAX_MESSAGE_LENGTH

import json

strategy_mapping = {
    "fault_tolerant_fedavg" : FaultTolerantFedAvg,
    "fedadagrad" : FedAdagrad,
    "fast_and_slow" : FastAndSlow,
    "fedavg" : FedAvg,
    "fedfs_v0" : FedFSv0,
    "fedfs_v1" : FedFSv1,
    "fedopt" : FedOpt,
    "qfedavg": QFedAvg
}

@click.group()
def server():
    pass

@server.command()
@click.option("-a","--server-address",type=str,default="[::]:8080",help="string (default: “[::]:8080”). The IPv6 address of the server.")
@click.option("-l","--grpc-max-message-length",type=int,default=GRPC_MAX_MESSAGE_LENGTH,help="int (default: 536_870_912, this equals 512MB). The maximum length of gRPC messages that can be exchanged with the Flower clients. The default should be sufficient for most models. Users who train very large models might need to increase this value. Note that the Flower clients need to be started with the same value (see flwr.client.start_client), otherwise clients will not know about the increased limit and block larger messages.")
@click.option("-f","--force-final-distributed-eval",type=bool,default=False,help="bool (default: False). Forces a distributed evaulation to occur after the last training epoch when enabled.")
@click.option("-s","--strategy",type=str,default="fedavg",help=f"{' | '.join(strategy_mapping.keys())} . Sets the strategy for Federated Learning.")
@click.option("-n",'--num-rounds',type=int,default=3,help="int (default: 3). Number of Rounds during Federated Learning")
@click.option("-p","--strategy-params",type=str,help="JSON. Sets the parameters for selected strategy. Parameters available are: fraction_fit: float, fraction_eval: float, min_fit_clients: int, min_eval_clients: int, min_available_clients:int, accept_failures: bool")

def start(server_address,grpc_max_message_length,force_final_distributed_eval,strategy,num_rounds, strategy_params):
    print(server_address,grpc_max_message_length,force_final_distributed_eval,strategy,num_rounds, strategy_params)
    if not strategy in strategy_mapping:
        raise ValueError("Value for strategy is invalid")
    
    selected_strategy = strategy_mapping[strategy]()

    if strategy_params:
        strategy_params = strategy_params.replace("'",'"')
        strategy_params = json.loads(strategy_params)
        selected_strategy = strategy_mapping[strategy](**strategy_params)

    start_server(
        server_address=server_address,
        grpc_max_message_length=grpc_max_message_length,
        force_final_distributed_eval=force_final_distributed_eval,
        strategy=selected_strategy,
        config={num_rounds:num_rounds}
    )