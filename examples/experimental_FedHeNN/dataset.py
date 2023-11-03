import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, RandomSampler, DataLoader
import warnings


use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)


def load_mnist_data_entire(batch_size=32):
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": batch_size}

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)
    num_examples = {"trainset": len(dataset1), "testset": len(dataset2)}
    return train_loader, test_loader, num_examples


def generate_custom_datasplits(partitions, input_seed):
    """Last slice corresponds to RAD with more datapoints
    TODO : hardcoded with 5 partitions"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_splits = random_split(
        dataset1,
        (10000, 10000, 10000, 10000, 20000),
        generator=torch.Generator().manual_seed(input_seed),
    )
    test_splits = random_split(
        dataset2,
        (1500, 1500, 1500, 1500, 4000),
        generator=torch.Generator().manual_seed(input_seed),
    )
    assert len(train_splits) == partitions, "Unequal train splits to partitions"
    assert len(test_splits) == partitions, "Unequal test splits to partitions"
    return train_splits, test_splits


def load_mnist_data_partition(
    batch_size=32,
    partitions=5,
    RAD=False,
    use_cuda=False,
    subsample_RAD=True,
    input_seed=100,
):
    """Last slice corresponds to RAD with more datapoints
    TODO : hardcoded with 5 partitions"""
    train_splits, test_splits = generate_custom_datasplits(partitions, input_seed)
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": batch_size}

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if RAD:
        if subsample_RAD:
            sample_frac = 0.1
        else:
            sample_frac = 1

        train_splits_subsampler = RandomSampler(
            train_splits[-1],
            num_samples=int(len(train_splits[-1]) * sample_frac),
            generator=torch.Generator().manual_seed(input_seed),
        )
        test_splits_subsampler = RandomSampler(
            test_splits[-1],
            num_samples=int(len(test_splits[-1]) * sample_frac),
            generator=torch.Generator().manual_seed(input_seed),
        )
        # batch size in RAD for entire dataset
        train_kwargs = {"batch_size": len(train_splits[-1])}
        test_kwargs = {"batch_size": len(test_splits[-1])}
        return (
            DataLoader(
                train_splits[-1], sampler=train_splits_subsampler, **train_kwargs
            ),
            DataLoader(test_splits[-1], sampler=test_splits_subsampler, **test_kwargs),
            {
                "trainset": len(train_splits_subsampler),
                "testset": len(test_splits_subsampler),
            },
        )

    return [
        (
            DataLoader(train_splits[part_idx], **train_kwargs),
            DataLoader(test_splits[part_idx], **test_kwargs),
            {
                "trainset": len(train_splits[part_idx]),
                "testset": len(test_splits[part_idx]),
            },
        )
        for part_idx in range(partitions - 1)
    ]


if __name__ == "__main__":
    data_loader = load_mnist_data_partition(RAD=True, subsample_RAD=True)[1]

    print(next(iter(data_loader))[1])
    print(load_mnist_data_partition(RAD=True, subsample_RAD=True)[2])
