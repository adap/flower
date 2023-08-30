from collections import defaultdict
from typing import Any, Dict

from flwr.server.strategy.strategy import Strategy


class StoreHistoryStrategy(Strategy):
    """Server FL history storage per training/evaluation round strategy
    implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hist: Dict[str, Dict[str, Any]] = {
            "trn": defaultdict(dict),
            "tst": defaultdict(dict),
        }
