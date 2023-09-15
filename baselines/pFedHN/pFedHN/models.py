## Have made the models more flexible referencing the original code implementation

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

###
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
####

class CNNHyper(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels, n_kernels,out_dim, hidden_dim=100,
            spec_norm=False, n_hidden=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        if self.in_channels == 3:

            self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
            self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
            self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
            self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
            self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)
            self.l1_bias = nn.Linear(hidden_dim, 120)
            self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
            self.l2_bias = nn.Linear(hidden_dim, 84)
            self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
            self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

        elif self.in_channels == 1:

            self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 3 * 3)
            self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
            self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 3 * 3)
            self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
            self.c3_weights = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels * 2 * self.n_kernels * 3 * 3)
            self.c3_bias = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels)
            self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * 2 * self.n_kernels * 3 * 3)
            self.l1_bias = nn.Linear(hidden_dim, 120)
            self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
            self.l2_bias = nn.Linear(hidden_dim, 84)
            self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
            self.l3_bias = nn.Linear(hidden_dim, self.out_dim)


        if spec_norm and self.in_channels == 3:

            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

        elif spec_norm and self.in_channels == 1:

            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.c3_weights = spectral_norm(self.c3_weights)
            self.c3_bias = spectral_norm(self.c3_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        if self.in_channels == 3 :

            weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2 * self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
            })

        elif self.in_channels == 1 :

            weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 3, 3),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 3, 3),
            "conv2.bias": self.c2_bias(features).view(-1),
            "conv3.weight": self.c3_weights(features).view(2 * 2 * self.n_kernels, 2 * self.n_kernels, 3, 3),
            "conv3.bias": self.c3_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2 * 2 * self.n_kernels * 3 * 3),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
            })

        return weights


class CNNTarget(nn.Module):
    def __init__(self, in_channels, n_kernels, out_dim):
        super(CNNTarget, self).__init__()

        self.in_channels = in_channels

        if self.in_channels == 3:
            self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
            self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, out_dim)
        
        elif self.in_channels == 1:

            self.conv1 = nn.Conv2d(in_channels, n_kernels, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 3)
            self.conv3 = nn.Conv2d(2 * n_kernels, 2 * 2 * n_kernels, 3)
            self.fc1 = nn.Linear(2 * 2 * n_kernels * 3 * 3, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        
        if self.in_channels == 3:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.in_channels == 1:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        return x
