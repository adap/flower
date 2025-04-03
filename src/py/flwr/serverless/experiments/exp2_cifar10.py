# Tensorflow logging level: warnings or higher
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.utils import set_random_seed
from subprocess import check_output
from experiments.utils.federated_learning_runner import FederatedLearningRunner


# main function
if __name__ == "__main__":
    # starts a new run
    from argparse import ArgumentParser
    from dotenv import load_dotenv

    load_dotenv()

    parser = ArgumentParser(
        description="Run federated learning experiments on CIFAR10."
    )

    # base config
    base_config = {
        "project": "cifar10",
        "epochs": 20,
        "batch_size": 128,
        "steps_per_epoch": 1200,
        "lr": 0.0005,
        "num_nodes": 2,
        "use_async": False,
        "federated_type": "concurrent",
        "dataset": "cifar10",
        "strategy": "fedavg",
        "data_split": "skewed",
        "skew_factor": 0.9,
        "test_steps": None,  # 50,
        "net": "resnet18",
        "random_seed": 0,
        "track": False,
    }
    for key, value in base_config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    parser.add_argument(
        "--use_default_configs", "-u", action="store_true", default=False
    )

    args = parser.parse_args()
    if args.use_default_configs:
        # Treatments
        # Single node (centralized) training.
        config_overides = [
            {
                "random_seed": random_seed,
                "num_nodes": 1,
            }
            for random_seed in [None, None]  # range(1, 3)
        ]
        config_overides += [
            {
                "random_seed": random_seed,
                "use_async": user_async,
                "skew_factor": skew_factor,
                "num_nodes": num_nodes,
                "strategy": strategy,
            }
            for random_seed in [100, 101]
            for user_async in [False]
            for skew_factor in [
                0,
                # 0.1,
                # 0.5,
                # 0.99,
                # 1,
                0.9,
            ]
            for num_nodes in [3, 5, 2]
            for strategy in [
                "fedavg",
                "fedavgm",
                # "fedadam",
            ]
        ]

    else:
        config_overide = {}
        for key, value in vars(args).items():
            config_overide[key] = value
        config_overides = [config_overide]

    for i, config_overide in enumerate(config_overides):
        config_overide["track"] = args.track
        config = {**base_config, **config_overide}
        print(
            f"\n***** Starting trial {i + 1} of {len(config_overides)} with config: {str(config)[:80]}...\n"
        )
        if args.use_default_configs:
            # use subprocess to run this script
            command = "python -m experiments.exp2_cifar10"
            for key, value in config_overide.items():
                if isinstance(value, bool):
                    if value:
                        command += f" --{key}"
                else:
                    command += f" --{key} {value}"
            print(command)
            # wait for the command to finish, stream to stdout
            check_output(command, shell=True)
        else:
            federated_learning_runner = FederatedLearningRunner(
                config=config,
            )
            federated_learning_runner.run()
