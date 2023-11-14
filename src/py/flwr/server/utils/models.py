import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchsummary import summary
import torchvision.models as models


class Generator(nn.Module):  # W_g
    def __init__(self, num_classes=7, latent_dim=256, other_dim=128):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, other_dim), nn.ReLU(), nn.BatchNorm1d(other_dim)
        )
        self.fc_mu = nn.Linear(other_dim, latent_dim)
        self.fc_logvar = nn.Linear(other_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y):
        z0 = self.net(y)
        mu = self.fc_mu(z0)
        logvar = self.fc_logvar(z0)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Enclassifier(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(Enclassifier, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Classifier
        self.clf = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.clf(z)
        return pred, mu, logvar


class AlexNet(nn.Module):
    def __init__(self, num_classes=7, latent_dim=4096, other_dim=1000):
        super(AlexNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)
        self.clf = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=False),
            # nn.Dropout(),
            nn.Linear(latent_dim, other_dim),
            nn.ReLU(inplace=False),
            nn.Linear(other_dim, num_classes),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.clf(z)
        return pred, mu, logvar


class ResNet18(nn.Module):
    def __init__(
        self, num_classes=7, latent_dim=256, other_dim=128, point_estimate=False
    ):
        super(ResNet18, self).__init__()
        # Load the pre-trained ResNet-18 model from torchvision
        self.resnet18 = models.resnet18(weights="DEFAULT")
        self.point_estimate = point_estimate
        # Extract individual layers for further modification if needed
        self.conv1 = self.resnet18.conv1
        self.bn1 = self.resnet18.bn1
        self.relu = self.resnet18.relu
        self.maxpool = self.resnet18.maxpool
        self.layer1 = self.resnet18.layer1
        self.layer2 = self.resnet18.layer2
        self.layer3 = self.resnet18.layer3
        self.layer4 = self.resnet18.layer4
        self.avgpool = self.resnet18.avgpool
        self.fc_mu = nn.Linear(self.resnet18.fc.in_features, latent_dim)
        self.fc_logvar = nn.Linear(self.resnet18.fc.in_features, latent_dim)

        # Modify self.fc to include two fully connected layers with ReLU and BatchNorm
        self.clf = nn.Sequential(
            nn.Linear(latent_dim, other_dim),
            nn.BatchNorm1d(other_dim),
            nn.ReLU(),
            nn.Linear(other_dim, num_classes),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        if self.point_estimate:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)

        pred = self.clf(z)
        return pred, mu, logvar


if __name__ == "__main__":
    net = ResNet18().to(DEVICE)
    summary(net, (3, 128, 128))
