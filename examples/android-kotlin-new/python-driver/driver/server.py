from logging import INFO

import flwr as fl
from flwr.common import GetParametersIns
from flwr.common.logger import log


class Server(fl.server.Server):
    def __init__(
        self,
        *,
        client_manager,
        model_id,
        strategy=None,
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.model_id = model_id

    def _get_initial_parameters(self, timeout):
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={"model_id": self.model_id})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters
