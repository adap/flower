# FedPara uses FedAvg as the default strategy
from flwr.server.strategy import FedAvg


class FedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
