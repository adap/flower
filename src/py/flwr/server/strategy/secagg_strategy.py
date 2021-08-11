from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from flwr.common.typing import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class SecAggStrategy(ABC):
    @abstractmethod
    def get_sec_agg_param(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def sec_agg_configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager, sample_num: int, min_num: int
    ) -> List[Tuple[ClientProxy, FitIns]]:
        pass
