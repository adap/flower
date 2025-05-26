import time
import logging
from uuid import uuid4
from typing import List, Tuple
from flwr.common import Parameters, Code, FitRes, Status
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from ..shared_folder.base_folder import SharedFolder
from .aggregatable import Aggregatable

LOGGER = logging.getLogger(__name__)

class AsyncFederatedNode:
    """Asynchronous federated learning node - doesn't wait for other nodes."""
    
    def __init__(self, shared_folder: SharedFolder, strategy: Strategy, node_id: str = None):
        self.node_id = node_id or str(uuid4())
        self.strategy = strategy
        self.model_store = shared_folder
        self.counter = 0
        self.seen_models = set()
    
    def update_parameters(self, local_parameters: Parameters, num_examples: int, 
                         metrics: dict = None, epoch: int = None) -> Tuple[Parameters, dict]:
        """Update parameters asynchronously - returns immediately with aggregated weights."""
        
        # Store local model
        self_aggregatable = Aggregatable(local_parameters, num_examples, metrics)
        model_hash = f"{self.node_id}_{time.time()}"
        self.model_store[self.node_id] = {
            'aggregatable': self_aggregatable,
            'model_hash': model_hash,
            'epoch': epoch,
            'node_id': self.node_id
        }
        
        # Get models from other nodes (non-blocking)
        other_models = self._get_other_models()
        
        if not other_models:
            return local_parameters, metrics
        
        # Aggregate with available models
        all_models = [self_aggregatable] + other_models
        return self._aggregate(all_models)
    
    def _get_other_models(self) -> List[Aggregatable]:
        """Get available models from other nodes."""
        other_models = []
        for key, value in self.model_store.items():
            if (key != self.node_id and isinstance(value, dict) and 
                'model_hash' in value):
                model_hash = value['model_hash']
                if model_hash not in self.seen_models:
                    self.seen_models.add(model_hash)
                    other_models.append(value['aggregatable'])
        return other_models
    
    def _aggregate(self, models: List[Aggregatable]) -> Tuple[Parameters, dict]:
        """Aggregate models using the strategy."""
        results = [
            (None, FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=model.parameters,
                num_examples=model.num_examples,
                metrics=model.metrics or {}
            ))
            for model in models
        ]
        
        aggregated_params, aggregated_metrics = self.strategy.aggregate_fit(
            server_round=self.counter + 1, results=results, failures=[]
        )
        self.counter += 1
        
        # Ensure we always have the correct node count and examples
        total_examples = sum(m.num_examples for m in models)
        num_nodes = len(models)
        
        # Simple metric aggregation if strategy didn't do it
        if not aggregated_metrics:
            aggregated_metrics = {}
        
        # Always include node count and total examples
        aggregated_metrics['num_examples'] = total_examples
        aggregated_metrics['num_nodes'] = num_nodes
        
        # Add weighted averages of other metrics if available
        if models[0].metrics:
            for key in models[0].metrics:
                if key not in ['num_nodes', 'num_examples']:
                    weighted_sum = sum(m.metrics[key] * m.num_examples for m in models if m.metrics and key in m.metrics)
                    if total_examples > 0:
                        aggregated_metrics[key] = weighted_sum / total_examples
        
        return aggregated_params, aggregated_metrics 