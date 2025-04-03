from flwr.serverless.experiments.utils.keras_federated_learning_runner import KerasFederatedLearningRunner


# main function
if __name__ == "__main__":
    # starts a new run
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
        "lr": 0.0005,
        "num_nodes": 2,
        "use_async": True,
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
        
        federated_learning_runner = KerasFederatedLearningRunner(
            config=config,
        )
        federated_learning_runner.run()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
