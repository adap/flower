from argparse import ArgumentParser

import numpy as np
import torch

import flwr as fl

import mnist

DATA_ROOT = "./data/mnist"
SAVE_MODEL_PATH = "./mnist_net.pth"

if __name__ == "__main__":
    # Training settings
    parser = ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--server_address",
        type=str,
        default="[::]:8080",
        help=f"gRPC server address (default: '[::]:8080')",
    )
    parser.add_argument(
        "--cid",
        type=int,
        default=0,
        metavar="N",
        help="ID of current client (default: 0)",
    )
    parser.add_argument(
        "--nb_clients",
        type=int,
        default=2,
        metavar="N",
        help="Total number of clients being launched (default: 2)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    for arg in vars(args):
        print (arg, getattr(args, arg))

    # Load MNIST data
    train_loader, test_loader = mnist.load_data(
        data_root=DATA_ROOT,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        cid=args.cid,
        nb_clients=args.nb_clients,
    )

    # pylint: disable=no-member
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pylint: enable=no-member

    # Instantiate client
    client = mnist.PytorchMNISTClient(
        cid = args.cid, train_loader=train_loader, test_loader=test_loader, epochs = args.epochs, device=device
    )

    # Start client
    fl.client.start_client(args.server_address, client)
