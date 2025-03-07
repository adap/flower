from datasets import load_dataset
import torchvision
from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def create_dataloaders(batch_size=256):
    use_dataset_stats_for_image_normalization = True

    transform_train_steps = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    transform_test_steps = [
        transforms.ToTensor(),
    ]

    if use_dataset_stats_for_image_normalization:
        step = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train_steps.append(step)
        transform_test_steps.append(step)

    transform_train = transforms.Compose(transform_train_steps)
    transform_test = transforms.Compose(transform_test_steps)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader


def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs = batch["img"].to(device)
                labels = batch["label"].to(device)
            else:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy}")
    return accuracy


if __name__ == "__main__":
    from net import SimpleConvNet, ResNet18
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomCrop

    ############################################################
    # Configuration
    ############################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    lr = 0.1
    max_steps = 200 * 36 # 50000
    log_every = 1000
    validate_every = 200
    num_partitions = 2
    ############################################################

    fds = FederatedDataset(
        dataset="uoft-cs/cifar10", partitioners={"train": IidPartitioner(num_partitions=num_partitions)}
    )

    # Load the first partition of the "train" split
    partition = fds.load_partition(0, "train")
    print(f"size of the partition in the training data: {len(partition)}")
    # transform the partition to a torch dataset
    train_transforms = Compose([
        ToTensor(),
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_transforms = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),        
    ])

    def apply_train_transforms(batch):
        # For CIFAR-10 the "img" column contains the images we want to apply the transforms to
        batch["img"] = [train_transforms(img) for img in batch["img"]]
        return batch
    
    def apply_val_transforms(batch):
        # For CIFAR-10 the "img" column contains the images we want to apply the transforms to
        batch["img"] = [val_transforms(img) for img in batch["img"]]
        return batch

    # TODO: for debugging, I am using the original training set
    # partition = load_dataset("uoft-cs/cifar10", split="train")
    partition_torch = partition.with_transform(apply_train_transforms)
    partition_torch = torch.utils.data.DataLoader(partition_torch, batch_size=batch_size, shuffle=True)
    
    # You can access the whole "test" split of the base dataset (it hasn't been partitioned)
    centralized_dataset = fds.load_split("test")
    # centralized_dataset = load_dataset("uoft-cs/cifar10", split="test")
    print(f"size of the centralized dataset in the test data: {len(centralized_dataset)}")
    # transform the centralized dataset to a torch dataset
    centralized_dataset_torch = centralized_dataset.with_transform(apply_val_transforms)
    centralized_dataset_torch = torch.utils.data.DataLoader(
        centralized_dataset_torch, batch_size=batch_size, shuffle=False)

    # train_loader, test_loader = create_dataloaders(batch_size)

    # A simple training loop
    model = ResNet18(small_resolution=True).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # add a scheduler
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, total_steps=max_steps)
    train_loader = partition_torch
    test_loader = centralized_dataset_torch

    step = 0
    while step < max_steps:
        for batch in train_loader: # partition_torch:
            # Get a batch of data
            if isinstance(batch, dict):
                inputs = batch["img"].to(device)
                labels = batch["label"].to(device)
            else:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
            assert inputs.shape[1:] == (3, 32, 32), f"The shape of the inputs is not correct. Got {inputs.shape[1:]}"

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            # clip the gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            ############################################################
            # Callbacks
            ############################################################
            if step % log_every == 0:
                print(f"Step {step} loss: {loss.item():.4f}, lr: {lr_scheduler.get_last_lr()[0]:.6f}")

            # Validate the model
            if step % validate_every == 0:
                # validate(model, centralized_dataset_torch, device)
                validate(model, test_loader, device)
            
            ############################################################
            # End of callbacks
            ############################################################
            
            step += 1
            if step >= max_steps:
                break
