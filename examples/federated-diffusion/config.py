import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--server_device", type=str, default="cpu")
    parser.add_argument("--nclass_cifar", type=int, default=2)
    parser.add_argument("--nsamples_cifar", type=int, default=int(50000 / 20))
    parser.add_argument("--rate_unbalance_cifar", type=float, default=1.0)
    parser.add_argument("--iid", action="store_true", default=False)
    parser.add_argument("--personalized", default=True)
    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_test", type=int, default=4)
    parser.add_argument("--personalization_layers", type=int, default=4)
    parser.add_argument("--personalization_path", type=str, default="docs/results/pers")
    parser.add_argument("--model_path", type=str, default="docs/results/models")
    parser.add_argument("--log_path", type=str, default="docs/results")

    # Parse arguments
    args = parser.parse_args()

    return args
