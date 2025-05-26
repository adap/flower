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

class SyncFederatedNode:
    """Synchronous federated learning node - waits for all nodes."""
    
    def __init__(self, shared_folder: SharedFolder, strategy: Strategy, 
                 num_nodes: int, node_id: str = None):
        self.node_id = node_id or str(uuid4())
        self.strategy = strategy
        self.model_store = shared_folder
        self.num_nodes = num_nodes
        self.counter = 0
    
    def update_parameters(self, local_parameters: Parameters, num_examples: int,
                         metrics: dict = None, epoch: int = None) -> Tuple[Parameters, dict]:
        """Update parameters synchronously - waits for all nodes."""
        
        # Store local model
        self_aggregatable = Aggregatable(local_parameters, num_examples, metrics)
        model_hash = f"{self.node_id}_{time.time()}"
        self.model_store[model_hash] = {
            'aggregatable': self_aggregatable,
            'model_hash': model_hash,
            'epoch': epoch,
            'node_id': self.node_id
        }
        
        # Wait for other nodes
        other_models = self._wait_for_other_nodes(epoch)
        
        # Aggregate all models
        all_models = other_models + [self_aggregatable]
        return self._aggregate(all_models)
    
    def _wait_for_other_nodes(self, epoch: int, max_wait: int = 300) -> List[Aggregatable]:
        """Wait for other nodes to submit their models."""
        wait_time = 0
        while wait_time < max_wait:
            other_models = []
            for key, value in self.model_store.items():
                if (isinstance(value, dict) and value.get('epoch') == epoch and 
                    value.get('node_id') != self.node_id):
                    other_models.append(value['aggregatable'])
            
            if len(other_models) >= self.num_nodes - 1:
                LOGGER.info(f"All {self.num_nodes} nodes ready for aggregation")
                return other_models
            
            LOGGER.info(f"Waiting for {self.num_nodes - 1 - len(other_models)} more nodes...")
            time.sleep(2)
            wait_time += 2
        
        LOGGER.warning(f"Timeout waiting for nodes, proceeding with {len(other_models)} nodes")
        return other_models
    
    def _aggregate(self, models: List[Aggregatable]) -> Tuple[Parameters, dict]:
        """Same aggregation logic as async node."""
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