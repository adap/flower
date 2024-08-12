"""$project_name: A Flower / FlowerTune app."""

from logging import INFO, WARN

from flwr.server.strategy import FedAvg
from flwr.common import log


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation."""
    def __init__(self, init_model, **kwargs):
        self.init_model = init_model
        super().__init__(**kwargs)

    def configure_fit(self, **kwargs):
        """Configure the next round of training."""
        return_clients = super().configure_fit(**kwargs)

        # Test communication costs
        num_clients = len(return_clients)
        test_communication_costs(self.init_model, num_clients)

        return return_clients


def test_communication_costs(model, num_clients):
    """Test communication costs per FL round."""

    trainable, _ = model.get_nb_trainable_parameters()
    comm_cost = 2 * num_clients * trainable / 1024**2
    log(INFO, f"Communication costs per round: {comm_cost} MB")

    if comm_cost > 500:
        log(WARN,
            "The total communication costs per round exceed 500 MB. "
            "Please consider reducing it if you plan to participate "
            "FlowerTune LLM Leaderboard.",
            )
