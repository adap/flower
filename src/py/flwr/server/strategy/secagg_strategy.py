from abc import ABC, abstractmethod
from typing import Dict


class SecAggStrategy(ABC):
    @abstractmethod
    def get_sec_agg_param(self) -> Dict[str, int]:
        pass
