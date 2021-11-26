"""federated learning with different aggregation strategy on benchmark exp."""
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import argparse
import copy
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from cnn_model import CNNModel
from torch import nn, optim
from utils import data_utils


def prepare_data(args):
    # Prepare data
    transform_mnist = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_svhn = transforms.Compose(
        [
            transforms.Resize([28, 28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_usps = transforms.Compose(
        [
            transforms.Resize([28, 28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_synth = transforms.Compose(
        [
            transforms.Resize([28, 28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_mnistm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # MNIST
    mnist_trainset = data_utils.DigitsDataset(
        data_path="data/MNIST",
        channels=1,
        percent=args.percent,
        train=True,
        transform=transform_mnist,
    )
    mnist_testset = data_utils.DigitsDataset(
        data_path="data/MNIST",
        channels=1,
        percent=args.percent,
        train=False,
        transform=transform_mnist,
    )

    # SVHN
    svhn_trainset = data_utils.DigitsDataset(
        data_path="data/SVHN",
        channels=3,
        percent=args.percent,
        train=True,
        transform=transform_svhn,
    )
    svhn_testset = data_utils.DigitsDataset(
        data_path="data/SVHN",
        channels=3,
        percent=args.percent,
        train=False,
        transform=transform_svhn,
    )

    # USPS
    usps_trainset = data_utils.DigitsDataset(
        data_path="data/USPS",
        channels=1,
        percent=args.percent,
        train=True,
        transform=transform_usps,
    )
    usps_testset = data_utils.DigitsDataset(
        data_path="data/USPS",
        channels=1,
        percent=args.percent,
        train=False,
        transform=transform_usps,
    )

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(
        data_path="data/SynthDigits/",
        channels=3,
        percent=args.percent,
        train=True,
        transform=transform_synth,
    )
    synth_testset = data_utils.DigitsDataset(
        data_path="data/SynthDigits/",
        channels=3,
        percent=args.percent,
        train=False,
        transform=transform_synth,
    )

    # MNIST-M
    mnistm_trainset = data_utils.DigitsDataset(
        data_path="data/MNIST_M/",
        channels=3,
        percent=args.percent,
        train=True,
        transform=transform_mnistm,
    )
    mnistm_testset = data_utils.DigitsDataset(
        data_path="data/MNIST_M/",
        channels=3,
        percent=args.percent,
        train=False,
        transform=transform_mnistm,
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        mnist_trainset, batch_size=args.batch, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        mnist_testset, batch_size=args.batch, shuffle=False
    )
    svhn_train_loader = torch.utils.data.DataLoader(
        svhn_trainset, batch_size=args.batch, shuffle=True
    )
    svhn_test_loader = torch.utils.data.DataLoader(
        svhn_testset, batch_size=args.batch, shuffle=False
    )
    usps_train_loader = torch.utils.data.DataLoader(
        usps_trainset, batch_size=args.batch, shuffle=True
    )
    usps_test_loader = torch.utils.data.DataLoader(
        usps_testset, batch_size=args.batch, shuffle=False
    )
    synth_train_loader = torch.utils.data.DataLoader(
        synth_trainset, batch_size=args.batch, shuffle=True
    )
    synth_test_loader = torch.utils.data.DataLoader(
        synth_testset, batch_size=args.batch, shuffle=False
    )
    mnistm_train_loader = torch.utils.data.DataLoader(
        mnistm_trainset, batch_size=args.batch, shuffle=True
    )
    mnistm_test_loader = torch.utils.data.DataLoader(
        mnistm_testset, batch_size=args.batch, shuffle=False
    )

    train_loaders = [
        mnist_train_loader,
        svhn_train_loader,
        usps_train_loader,
        synth_train_loader,
        mnistm_train_loader,
    ]
    test_loaders = [
        mnist_test_loader,
        svhn_test_loader,
        usps_test_loader,
        synth_test_loader,
        mnistm_test_loader,
    ]

    return train_loaders, test_loaders


def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step > 0:
            w_diff = torch.tensor(0.0, device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2.0 * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output = model(data)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / len(test_loader), correct / len(test_loader.dataset)


################# Key Function ########################
def communication(args, server_model, models, client_weights, client_num):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == "fedbn":
            for key in server_model.state_dict().keys():
                if "bn" not in key:
                    temp = torch.zeros_like(
                        server_model.state_dict()[key], dtype=torch.float32
                    )
                    for client_idx in range(client_num):
                        temp += (
                            client_weights[client_idx]
                            * models[client_idx].state_dict()[key]
                        )
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(
                        models[0].state_dict()[key]
                    )
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += (
                            client_weights[client_idx]
                            * models[client_idx].state_dict()[key]
                        )
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )
    return server_model, models


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Device:", device)
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="whether to make a log")
    parser.add_argument("--test", action="store_true", help="test the pretrained model")
    parser.add_argument(
        "--percent", type=float, default=0.1, help="percentage of dataset to train"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument(
        "--iters", type=int, default=100, help="iterations for communication"
    )
    parser.add_argument(
        "--wk_iters",
        type=int,
        default=1,
        help="optimization iters in local worker between communication",
    )
    parser.add_argument(
        "--mode", type=str, default="fedbn", help="fedavg | fedprox | fedbn"
    )
    parser.add_argument(
        "--mu", type=float, default=1e-2, help="The hyper parameter for fedprox"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../checkpoint/digits",
        help="path to save the checkpoint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from the save path checkpoint",
    )
    args = parser.parse_args()

    print("Export federated digits")
    exp_folder = "federated_digits"

    args.save_path = os.path.join(args.save_path, exp_folder)

    log = args.log
    if log:
        log_path = os.path.join("../logs/digits/", exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, "{}.log".format(args.mode)), "a")
        logfile.write(
            "==={}===\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        )
        logfile.write("===Setting===\n")
        logfile.write("    lr: {}\n".format(args.lr))
        logfile.write("    batch: {}\n".format(args.batch))
        logfile.write("    iters: {}\n".format(args.iters))
        logfile.write("    wk_iters: {}\n".format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, "{}".format(args.mode))

    print("Model initialization")
    server_model = CNNModel().to(device)
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    train_loaders, test_loaders = prepare_data(args)

    # name of each client dataset
    datasets = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]

    # federated setting
    client_num = len(datasets)
    client_weights = [1 / client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.test:
        print("Loading snapshots...")
        checkpoint = torch.load("../snapshots/digits/{}".format(args.mode.lower()))
        server_model.load_state_dict(checkpoint["server_model"])
        if args.mode.lower() == "fedbn":
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(
                    checkpoint["model_{}".format(client_idx)]
                )
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print(
                    " {:<11s}| Test  Acc: {:.4f}".format(datasets[test_idx], test_acc)
                )
        else:
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(
                    " {:<11s}| Test  Acc: {:.4f}".format(datasets[test_idx], test_acc)
                )
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint["server_model"])
        if args.mode.lower() == "fedbn":
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(
                    checkpoint["model_{}".format(client_idx)]
                )
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint["server_model"])
        resume_iter = int(checkpoint["a_iter"]) + 1
        print("Resume training from epoch {}".format(resume_iter))
    else:
        resume_iter = 0

    # start training
    for a_iter in range(resume_iter, args.iters):
        optimizers = [
            optim.SGD(params=models[idx].parameters(), lr=args.lr)
            for idx in range(client_num)
        ]
        for wi in range(args.wk_iters):
            print(
                "============ Train epoch {} ============".format(
                    wi + a_iter * args.wk_iters
                )
            )
            if args.log:
                logfile.write(
                    "============ Train epoch {} ============\n".format(
                        wi + a_iter * args.wk_iters
                    )
                )

            for client_idx in range(client_num):
                model, train_loader, optimizer = (
                    models[client_idx],
                    train_loaders[client_idx],
                    optimizers[client_idx],
                )
                if args.mode.lower() == "fedprox":
                    if a_iter > 0:
                        train_fedprox(
                            args,
                            model,
                            train_loader,
                            optimizer,
                            loss_fun,
                            client_num,
                            device,
                        )
                    else:
                        train(
                            model, train_loader, optimizer, loss_fun, client_num, device
                        )
                else:
                    train(model, train_loader, optimizer, loss_fun, client_num, device)

        # aggregation
        server_model, models = communication(
            args, server_model, models, client_weights, client_num
        )

        # report after aggregation
        for client_idx in range(client_num):
            model, train_loader, optimizer = (
                models[client_idx],
                train_loaders[client_idx],
                optimizers[client_idx],
            )
            train_loss, train_acc = test(model, train_loader, loss_fun, device)
            print(
                " {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}".format(
                    datasets[client_idx], train_loss, train_acc
                )
            )
            if args.log:
                logfile.write(
                    " {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n".format(
                        datasets[client_idx], train_loss, train_acc
                    )
                )
        # start testing
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(
                " {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}".format(
                    datasets[test_idx], test_loss, test_acc
                )
            )
            if args.log:
                logfile.write(
                    " {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n".format(
                        datasets[test_idx], test_loss, test_acc
                    )
                )

    # Save checkpoint
    print(" Saving checkpoints to {}...".format(SAVE_PATH))
    if args.mode.lower() == "fedbn":
        torch.save(
            {
                "model_0": models[0].state_dict(),
                "model_1": models[1].state_dict(),
                "model_2": models[2].state_dict(),
                "model_3": models[3].state_dict(),
                "model_4": models[4].state_dict(),
                "server_model": server_model.state_dict(),
            },
            SAVE_PATH,
        )
    else:
        torch.save(
            {
                "server_model": server_model.state_dict(),
            },
            SAVE_PATH,
        )

    if log:
        logfile.flush()
        logfile.close()


if __name__ == "__main__":
    main()
