from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import signal
from typing import List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import (
    FedAvg,
    FedAdam,
    FedAvgM,
    FedOpt,
    FedMedian,
)

from flwr.serverless.federated_node.async_federated_node import AsyncFederatedNode
from flwr.serverless.federated_node.sync_federated_node import SyncFederatedNode
from flwr.serverless.shared_folder.in_memory_folder import InMemoryFolder
from flwr.serverless.experiments.utils.base_experiment_runner import BaseExperimentRunner, Config


class TorchFederatedLearningRunner(BaseExperimentRunner):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.storage_backend: Any = InMemoryFolder()
        # In one round, each node trains on its local data for one epoch.
        self.num_rounds = self.epochs  # number of federated rounds (similar to epochs)
        self.lag = 0.1  # lag between nodes in pseudo-concurrent mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.should_stop = False
        
        def signal_handler(signum, frame):
            print("\nReceived interrupt signal. Shutting down gracefully...")
            self.should_stop = True
            
        self.original_handler = signal.signal(signal.SIGINT, signal_handler)

    def __del__(self):
        # Restore original signal handler when the object is destroyed
        signal.signal(signal.SIGINT, self.original_handler)

    def run(self):
        try:
            config: Config = self.config
            print("\n====== Starting Federated Learning Experiment ======")
            print(f"Configuration:\n- Nodes: {self.config.num_nodes}")
            print(f"- Strategy: {self.config.strategy}")
            print(f"- Mode: {'Asynchronous' if self.config.use_async else 'Synchronous'}")
            print(f"- Data split: {self.config.data_split}")
            print(f"- Rounds: {self.num_rounds}")
            print(f"- Device: {self.device}")
            
            if config.random_seed is not None:
                print(f"- Random seed: {config.random_seed}")
                torch.manual_seed(config.random_seed)

            if config.track:
                import wandb
                strategy = self.config.strategy
                num_nodes = self.config.num_nodes
                data_split = self.config.data_split
                sync_or_async: str = "async" if self.config.use_async else "sync"
                name = f"{sync_or_async}_{strategy}_{num_nodes}_nodes_{data_split}"
                if data_split == "skewed":
                    name += f"_{self.config.skew_factor}"
                wandb.init(
                    project=self.config.project,
                    entity=os.getenv("WANDB_ENTITY", "example_entity"),
                    name=name,
                    config=config.__dict__,
                )
            
            self.models = self.create_models()
            self.set_strategy()
            (
                self.partitioned_x_train,
                self.partitioned_y_train,
                self.x_test,
                self.y_test,
            ) = self.split_data()
            print("x_test shape:", self.x_test.shape)
            print("y_test shape:", self.y_test.shape)
            self.train_federated_models()
            if not self.should_stop:
                self.evaluate()
            if config.track:
                wandb.finish()
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, self.original_handler)

    def set_strategy(self):
        if self.strategy_name == "fedavg":
            self.strategies = [FedAvg() for _ in range(self.num_nodes)]
        elif self.strategy_name == "fedavgm":
            self.strategies = [FedAvgM() for _ in range(self.num_nodes)]
        elif self.strategy_name == "fedadam":
            self.strategies = [
                FedAdam(
                    initial_parameters=ndarrays_to_parameters(
                        [p.cpu().numpy() for p in self.models[i].parameters()]
                    )
                )
                for i in range(self.num_nodes)
            ]
        elif self.strategy_name == "fedopt":
            self.strategies = [
                FedOpt(
                    initial_parameters=ndarrays_to_parameters(
                        [p.cpu().numpy() for p in self.models[i].parameters()]
                    )
                )
                for i in range(self.num_nodes)
            ]
        elif self.strategy_name == "fedmedian":
            self.strategies = [FedMedian() for _ in range(self.num_nodes)]
        else:
            raise ValueError(f"Strategy not supported: {self.strategy_name}")

    def split_data(self):
        config: Config = self.config
        print("\n----- Data Splitting Phase -----")
        print(f"Splitting data using '{self.data_split}' strategy")
        
        if self.data_split == "random":
            print("Randomly distributing data across nodes")
            return self.random_split()
        elif self.data_split == "partitioned":
            print("Creating partitioned datasets with natural distribution")
            return self.create_partitioned_datasets()
        elif self.data_split == "skewed":
            print(f"Creating skewed partition with factor {config.skew_factor}")
            return self.create_skewed_partition_split(skew_factor=config.skew_factor)
        else:
            raise ValueError("Data split not supported")
        
    def get_train_dataloader_for_node(self, node_idx: int):
        partition_idx = node_idx
        partitioned_x_train = self.partitioned_x_train
        partitioned_y_train = self.partitioned_y_train
        # create a torch dataloader from the partitioned data
        x_train = torch.from_numpy(partitioned_x_train[partition_idx])
        # make the images channel first
        x_train = x_train.permute(0, 3, 1, 2)
        y_train = torch.from_numpy(partitioned_y_train[partition_idx])
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
    
    def get_test_dataloader(self):
        x_test = torch.from_numpy(self.x_test)
        x_test = x_test.permute(0, 3, 1, 2)
        y_test = torch.from_numpy(self.y_test)
        dataset = torch.utils.data.TensorDataset(x_test, y_test)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def train_federated_models(self) -> List[nn.Module]:
        print("\n----- Starting Federated Training -----")
        print(f"Training type: {self.federated_type}")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Batch size: {self.batch_size}")

        if self.federated_type == "pseudo-concurrent":
            print("Training federated models pseudo-concurrently with lag:", self.lag)
            return self._train_federated_models_pseudo_concurrently(self.models)
        elif self.federated_type == "concurrent":
            print("Training federated models concurrently using ThreadPoolExecutor")
            return self._train_federated_models_concurrently(self.models)
        else:
            print("Training federated models sequentially")
            return self._train_federated_models_sequentially(self.models)

    def _train_federated_models_concurrently(
        self, model_federated: List[nn.Module]
    ) -> List[nn.Module]:
        nodes = self.create_nodes()
        num_partitions = self.num_nodes
        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        try:
            with ThreadPoolExecutor(max_workers=self.num_nodes) as ex:
                futures = []
                for i_node in range(self.num_nodes):
                    if self.should_stop:
                        break
                    model = model_federated[i_node].to(self.device)
                    future = ex.submit(
                        self._train_node,
                        model=model,
                        train_loader=train_loaders[i_node],
                        node=nodes[i_node],
                        node_id=i_node,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    if self.should_stop:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in training thread: {e}")
                        self.should_stop = True
                        break

        finally:
            # Shutdown the executor
            ex.shutdown(wait=False)

        return model_federated

    def _train_federated_models_pseudo_concurrently(
        self, model_federated: List[nn.Module]
    ) -> List[nn.Module]:
        nodes = self.create_nodes()
        num_partitions = self.num_nodes

        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        seqs = [[]] * self.num_nodes
        for i_node in range(self.num_nodes):
            seqs[i_node] = [
                (i_node, j + i_node * self.lag) for j in range(self.num_rounds)
            ]
        # mix them up
        execution_sequence = []
        for i_node in range(self.num_nodes):
            execution_sequence.extend(seqs[i_node])
        execution_sequence = [
            x[0] for x in sorted(execution_sequence, key=lambda x: x[1])
        ]
        print(f"Execution sequence: {execution_sequence}")

        for i_node in execution_sequence:
            print("Training node", i_node, "is running...")
            model = model_federated[i_node].to(self.device)
            self._train_node(
                model=model,
                train_loader=train_loaders[i_node],
                node=nodes[i_node],
                node_id=i_node,
            )

        return model_federated

    def _train_federated_models_sequentially(
        self, model_federated: List[nn.Module]
    ) -> List[nn.Module]:
        nodes = self.create_nodes()
        num_partitions = self.num_nodes
        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        for i_round in range(self.num_rounds):
            print("\n============ Round", i_round)
            for i_partition in range(num_partitions):
                model = model_federated[i_partition].to(self.device)
                self._train_node(
                    model=model,
                    train_loader=train_loaders[i_partition],
                    node=nodes[i_partition],
                    node_id=i_partition,
                )

        return model_federated

    def _train_node(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        node: Union[AsyncFederatedNode, SyncFederatedNode],
        node_id: int,
    ):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.num_rounds):
            if self.should_stop:
                break
                
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
                if self.should_stop:
                    break
                    
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if i % 100 == 99:
                    accuracy = correct / total
                    print(f'[Node {node_id}] Epoch [{epoch + 1}/{self.num_rounds}], '
                          f'Step [{i + 1}/{len(train_loader)}], '
                          f'Loss: {running_loss / 100:.3f}, '
                          f'Acc: {100. * accuracy:.3f}%')
                    running_loss = 0.0

                if self.config.track:
                    import wandb
                    accuracy = correct / total
                    wandb.log({
                        f"node_{node_id}/loss": loss.item(),
                        f"node_{node_id}/accuracy": accuracy,
                    })

            ### Perform federation using the node.
            # First, convert torch model weights to flwr parameters.
            flwr_parameters = ndarrays_to_parameters([p.detach().cpu().numpy() for p in model.parameters()])
            # Then, perform federation using the node.
            accuracy = correct / total
            updated_parameters, metrics = node.update_parameters(
                local_parameters=flwr_parameters,
                num_examples=len(train_loader),
                metrics={
                    "loss": loss.item(),
                    "accuracy": accuracy,
                },
                epoch=epoch,
            )
            # Convert flwr parameters back to torch model weights.
            np_parameters = parameters_to_ndarrays(updated_parameters)
            for i, param in enumerate(model.parameters()):
                param.data = torch.from_numpy(np_parameters[i]).to(self.device)

    def create_nodes(self):
        if self.use_async:
            nodes = [
                AsyncFederatedNode(
                    shared_folder=self.storage_backend, strategy=self.strategies[i]
                )
                for i in range(self.num_nodes)
            ]
        else:
            nodes = [
                SyncFederatedNode(
                    shared_folder=self.storage_backend,
                    strategy=self.strategies[i],
                    num_nodes=self.num_nodes,
                )
                for i in range(self.num_nodes)
            ]
        return nodes

    def evaluate(self):
        print("\n----- Final Evaluation Phase -----")
        for i_node in [0]:  # range(self.num_nodes):
            print(f"Evaluating model from node {i_node} on test set:")
            model = self.models[i_node].to(self.device)
            model.eval()
            
            correct = 0
            total = 0
            test_loss = 0
            criterion = nn.CrossEntropyLoss()

            with torch.no_grad():
                for inputs, labels in self.get_test_dataloader():
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            accuracy = correct / total
            avg_loss = test_loss / len(self.get_test_dataloader())
            
            print(f"Test Loss: {avg_loss:.4f}")
            print(f"Test Accuracy: {100. * accuracy:.4f}%")

            if self.config.track:
                import wandb
                to_log = {
                    "test_accuracy": accuracy,
                    "test_loss": avg_loss,
                }
                wandb.log(to_log)
        print("\n====== Federated Learning Experiment Complete ======")
