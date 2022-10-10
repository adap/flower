from typing import List, Optional, Tuple
from collections import OrderedDict
import glob
import os

import torch
import flwr as fl

import cifar

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load PyTorch model
net = cifar.Net().to(DEVICE)
continue_from_checkpoint = True 
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):

        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        # log_dict['aggregated_parameters']=aggregated_parameters
    
        if aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(net.state_dict(), f"model_round_{rnd}.pth")

        return aggregated_parameters_tuple

if continue_from_checkpoint:
    def load_parameters_from_disk():
        # import Net
        list_of_files = [fname for fname in glob.glob("./model_round_*")]
        latest_round_file = max(list_of_files, key=os.path.getctime)
        print("Loading pre-trained model from: ", latest_round_file)
        state_dict = torch.load(latest_round_file)
        net.load_state_dict(state_dict)
        weights = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        return fl.common.ndarrays_to_parameters(weights)
        
strategy = SaveModelStrategy(
    initial_parameters=load_parameters_from_disk() if continue_from_checkpoint else None
)


if __name__ == "__main__":
    fl.server.start_server(server_address="0.0.0.0:8080", 
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=3))