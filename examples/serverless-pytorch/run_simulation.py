from flwr.serverless.experiments.utils.torch_federated_learning_runner import TorchFederatedLearningRunner
import torch
from net import ResNet18

# TODO: switch to flwr_datasets to handle data partitioning


class CIFAR10ExperimentRunner(TorchFederatedLearningRunner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_models(self):
        """Create PyTorch models for each node."""
        try:
            models = [ResNet18(small_resolution=True).to(self.device) for _ in range(self.num_nodes)]
            # Initialize models with same weights
            if len(models) > 1:
                for model in models[1:]:
                    model.load_state_dict(models[0].state_dict())
            return models
        except Exception as e:
            raise RuntimeError(f"Failed to create models: {str(e)}")

# main function
if __name__ == "__main__":
    import time
    from argparse import ArgumentParser
    
    start_time = time.time()

    parser = ArgumentParser(
        description="Run federated learning experiments on CIFAR10."
    )

    # base config
    base_config = {
        "project": "cifar10",
        "epochs": 20,
        "batch_size": 128,
        "steps_per_epoch": 1200,
        "lr": 0.1,
        "num_nodes": 2,
        "use_async": True,
        "federated_type": "concurrent",
        "dataset": "cifar10",
        "strategy": "fedavg",
        "data_split": "skewed",
        "skew_factor": 0.5,
        "test_steps": None,
        "net": "resnet18",
        "random_seed": 0,
        "track": False,
    }

    # Add arguments based on base config
    for key, value in base_config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    args = parser.parse_args()

    # Update config with any command line args
    config = {**base_config}
    for key, value in vars(args).items():
        config[key] = value

    print(f"\n***** Starting experiment with config: {str(config)[:80]}...\n")
    
    federated_learning_runner = CIFAR10ExperimentRunner(
        config=config,
    )
    federated_learning_runner.run()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
