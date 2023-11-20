from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import Net
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

import flwr as fl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data_mnist():
    """Load MNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = MNIST(root="./data", train=True, download=True, transform=transform)
    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, testloader


def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, testloader


def query_sample():
    _, mnist_test_loader = load_data()

    for idx, data in enumerate(mnist_test_loader):
        images = data[0].to(DEVICE)
        break  # Only take the first batch

    img_tensor = transforms.ToPILImage()(images[0])
    img_tensor.save("query.png")
    return images[0].unsqueeze(0)


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, _ in trainloader:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            loss.backward()
            optimizer.step()
    return net


def test(net, testloader):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            images = data[0].to(DEVICE)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)

    return loss.item() / total


def test2(net, testloader, gen=False, rnd=None):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            images = data[0].to(DEVICE)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)
        if gen:
            # Take one image from the last batch
            image_to_generate = images[0].unsqueeze(0)

            # Generate image using your generate function
            generated_tensors = generate(net, image_to_generate)
            generated_img = generated_tensors[0]
            # Image dimensions: 1, 3, 32, 32
            # RGB image for PIL
            generated_img = generated_img.squeeze(0)
            img_tensor = transforms.ToPILImage()(generated_img)
            img_tensor.save(f"output_{rnd}.png")
    return loss.item() / total


def sample(net):
    """Generates samples using the decoder of the trained VAE."""
    with torch.no_grad():
        z = torch.randn(10)
        z = z.to(DEVICE)
        gen_image = net.decode(z)
    return gen_image


def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)


def main():
    # Load model and data
    net = Net()
    net = net.to(DEVICE)
    trainloader, testloader = load_data()

    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("inside fit")
            print(config)
            new_net = train(net, trainloader, epochs=10)
            output_img = generate(new_net, query_sample())[0]
            output_img = output_img.squeeze(0)
            img_tensor = transforms.ToPILImage()(output_img)
            img_tensor.save(f"local_output_{config['server_round']}.png")

            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss = test(net, testloader)
            return float(loss), len(testloader), {}

    fl.client.start_client(
        server_address="127.0.0.1:8080", client=CifarClient().to_client()
    )


if __name__ == "__main__":
    main()
    # net = Net()
    # new_net = net.to(DEVICE)
    # output_img = generate(new_net, query_sample())[0]
    # output_img = output_img.squeeze(0)
    # img_tensor = transforms.ToPILImage()(output_img)
    # img_tensor.save(f"local_output.png")
