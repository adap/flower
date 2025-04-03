from dataclasses import dataclass
from flwr.common import Parameters


@dataclass
class Aggregatable:
    parameters: Parameters
    num_examples: int
    metrics: dict
