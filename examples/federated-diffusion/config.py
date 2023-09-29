from dataclasses import dataclass


@dataclass
class Params:
    num_inference_steps: int = 100
    num_clients: int = 2
    num_epochs: int = 1
    num_rounds: int = 2
    server_device: str = "cpu"
    nclass_cifar: int = 2
    nsamples_cifar: int = int(50000 / 20)
    rate_unbalance_cifar: float = 1.0
    iid: bool = False
    personalized: bool = True


PARAMS = Params()
