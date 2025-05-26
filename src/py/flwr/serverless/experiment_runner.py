from concurrent.futures import ThreadPoolExecutor
import logging
import signal
import sys
from typing import List, Any, Callable, Dict, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from flwr.server.strategy import FedAvg
from .federated_node.async_node import AsyncFederatedNode
from .federated_node.sync_node import SyncFederatedNode
from .shared_folder.in_memory_folder import InMemoryFolder
from .data_splitter import DataSplitter


LOGGER = logging.getLogger(__name__)


class FederatedExperimentRunner:
    """Experiment runner with centralized evaluation, test data management, and skewed data distribution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_nodes = config.get("num_nodes", 2)
        self.epochs = config.get("epochs", 5)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.use_async = config.get("use_async", True)
        self.train_fraction = config.get("train_fraction", 0.9)
        self.storage = InMemoryFolder()
        self.strategy = FedAvg()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data distribution settings
        self.data_split = config.get("data_split", "random")  # "random", "skewed"
        self.skew_factor = config.get("skew_factor", 0.8)  # 0.5 = random, 1.0 = completely skewed
        self.data_splitter = DataSplitter(strategy=self.data_split, skew_factor=self.skew_factor)
        
        # Test data management
        self.test_loader: Optional[DataLoader] = None
        self.evaluation_results = []  # Store centralized evaluation results
        self.node_evaluation_results = []  # Store per-node evaluation results
        
        # Graceful shutdown handling
        self.interrupted = False
        self.executor = None
        self.partial_results = {}
        
        # Create a shared interruption flag that can be accessed by training threads
        import threading
        self.interruption_event = threading.Event()
        
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print(f"\n{'='*60}")
        print("🛑 Interrupt signal received (Ctrl+C)")
        print("Attempting graceful shutdown...")
        print("Please wait while we clean up resources...")
        print(f"{'='*60}")
        
        self.interrupted = True
        self.interruption_event.set()  # Signal all training threads to stop
        
        # If we have an active executor, try to shutdown gracefully
        if self.executor:
            print("Shutting down thread pool executor...")
            self.executor.shutdown(wait=False)
        
        print("Graceful shutdown completed.")
        sys.exit(0)
    
    def set_test_data(self, test_loader: DataLoader):
        """Set the test data for centralized evaluation."""
        self.test_loader = test_loader
        print(f"Test data set with {len(test_loader.dataset)} samples")
    
    def create_data_partitions(self, dataset, num_partitions: int, num_classes: int = 10) -> List[Subset]:
        """Create data partitions based on the configured split strategy."""
        return self.data_splitter.split(dataset, num_partitions, num_classes)
    
    def create_nodes(self) -> List[Any]:
        """Create federated nodes based on configuration."""
        nodes = []
        for i in range(self.num_nodes):
            if self.use_async:
                node = AsyncFederatedNode(
                    shared_folder=self.storage,
                    strategy=self.strategy,
                    node_id=f"node_{i}"
                )
            else:
                node = SyncFederatedNode(
                    shared_folder=self.storage,
                    strategy=self.strategy,
                    num_nodes=self.num_nodes,
                    node_id=f"node_{i}"
                )
            nodes.append(node)
        return nodes
    
    def run_experiment(self, 
                      model_factory: Callable[[], Any],
                      train_dataset: Any,
                      train_fn: Callable[[Any, DataLoader, Any, dict, Optional[DataLoader]], Tuple[dict, Any]],
                      test_loader: Optional[DataLoader] = None,
                      num_classes: int = 10):
        """Run the federated learning experiment with centralized evaluation and data partitioning."""
        
        # Set test data if provided
        if test_loader is not None:
            self.set_test_data(test_loader)
        
        print(f"Starting {'async' if self.use_async else 'sync'} FL experiment")
        print(f"Nodes: {self.num_nodes}, Epochs: {self.epochs}")
        print(f"Data split: {self.data_split}" + (f" (skew factor: {self.skew_factor})" if self.data_split == "skewed" else ""))
        print(f"Centralized evaluation: {'Enabled' if self.test_loader else 'Disabled'}")
        
        # Create data partitions
        print(f"\n=== Data Partitioning ===")
        print(f"  Training examples: {len(train_dataset)}")
        partitions = self.create_data_partitions(train_dataset, self.num_nodes, num_classes)
        
        # Create data loaders for each partition
        data_loaders = []
        for i, partition in enumerate(partitions):
            # Split each partition into train/val
            train_size = int(self.train_fraction * len(partition))
            
            # Create a shuffled copy of indices (we do want the model to see all examples in the training set)
            shuffled_indices = partition.indices.copy()
            np.random.shuffle(shuffled_indices)
            train_indices = shuffled_indices[:train_size]
            val_indices = shuffled_indices[train_size:]
            
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)
            
            trainloader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            valloader = DataLoader(val_subset, batch_size=self.batch_size)
            
            data_loaders.append((trainloader, valloader))
            print(f"Node {i}: {len(train_subset)} train, {len(val_subset)} val samples")
        
        # Create nodes and models
        nodes = self.create_nodes()
        models = [model_factory().to(self.device) for _ in range(self.num_nodes)]
        
        # Training phase - each node trains for full epochs with per-epoch federated aggregation
        print(f"\n=== Training and Federated Aggregation Phase ===")
        print(f"Each node will train for {self.epochs} epochs with federated aggregation after each epoch")
        print(f"Only node 0 will perform centralized evaluation after each epoch")
        
        # Run training concurrently using ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=self.num_nodes)
        try:
            futures = []
            for node_idx, (node, model, dataloader) in enumerate(zip(nodes, models, data_loaders)):
                # Check for interruption before starting each node
                if self.interrupted:
                    print("Training interrupted before starting all nodes")
                    break
                
                # Only pass test_loader to node 0 for centralized evaluation
                test_loader_for_node = self.test_loader if node_idx == 0 else None
                
                # Create a config copy with interruption event
                node_config = self.config.copy()
                node_config['_interruption_event'] = self.interruption_event
                
                future = self.executor.submit(
                    train_fn, model, dataloader, node, node_config, test_loader_for_node
                )
                futures.append((node_idx, future))
            
            # Wait for all nodes to complete training and aggregation
            training_metrics = {}
            final_models = {}
            completed_nodes = 0
            
            for node_idx, future in futures:
                if self.interrupted:
                    print(f"Training interrupted. Completed {completed_nodes}/{len(futures)} nodes")
                    break
                
                try:
                    # Use a timeout to allow checking for interruption
                    final_model = future.result(timeout=None)
                    final_models[node_idx] = final_model
                    completed_nodes += 1
                    print(f"\nNode {node_idx} completed all epochs with federated aggregation")
                    
                except Exception as e:
                    if not self.interrupted:
                        print(f"Node {node_idx} failed with error: {e}")
                    self.partial_results[f"node_{node_idx}"] = {
                        'completed': False,
                        'error': str(e)
                    }
        
        finally:
            # Clean up executor
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
        
        print("Experiment completed!")
        return models, nodes
    