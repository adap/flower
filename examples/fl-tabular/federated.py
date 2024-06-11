from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim

import numpy as np
import random

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

NUMBER_OF_CLIENTS = 5

categories = [
    [
        "Private",
        "?",
        "Self-emp-not-inc",
        "Federal-gov",
        "Self-emp-inc",
        "Local-gov",
        "State-gov",
        "Never-worked",
        "Without-pay",
    ],
    [
        "Some-college",
        "HS-grad",
        "Assoc-voc",
        "Bachelors",
        "7th-8th",
        "Prof-school",
        "10th",
        "Doctorate",
        "11th",
        "12th",
        "Masters",
        "9th",
        "Preschool",
        "Assoc-acdm",
        "1st-4th",
        "5th-6th",
    ],
    [
        "Married-civ-spouse",
        "Never-married",
        "Divorced",
        "Widowed",
        "Married-spouse-absent",
        "Separated",
        "Married-AF-spouse",
    ],
    [
        "Transport-moving",
        "?",
        "Adm-clerical",
        "Machine-op-inspct",
        "Tech-support",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Craft-repair",
        "Farming-fishing",
        "Other-service",
        "Protective-serv",
        "Priv-house-serv",
        "Armed-Forces",
    ],
    ["Husband", "Own-child", "Not-in-family", "Unmarried", "Other-relative", "Wife"],
    ["White", "Asian-Pac-Islander", "Black", "Other", "Amer-Indian-Eskimo"],
    ["Male", "Female"],
    [
        "United-States",
        "Philippines",
        "Canada",
        "Germany",
        "Vietnam",
        "Italy",
        "Jamaica",
        "Cuba",
        "Columbia",
        "Taiwan",
        "Mexico",
        "Japan",
        "?",
        "Ireland",
        "Puerto-Rico",
        "India",
        "Iran",
        "El-Salvador",
        "Guatemala",
        "Haiti",
        "Outlying-US(Guam-USVI-etc)",
        "China",
        "Portugal",
        "England",
        "South",
        "France",
        "Honduras",
        "Dominican-Republic",
        "Cambodia",
        "Hong",
        "Greece",
        "Nicaragua",
        "Ecuador",
        "Poland",
        "Peru",
        "Laos",
        "Trinadad&Tobago",
        "Yugoslavia",
        "Thailand",
        "Scotland",
        "Holand-Netherlands",
        "Hungary",
    ],
]


def load_data(partition_id):
    # Load the FederatedDataset
    fds = FederatedDataset(
        dataset="scikit-learn/adult-census-income",
        partitioners={"train": NUMBER_OF_CLIENTS},
    )

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    dataset["income"] = dataset["income"].apply(lambda x: 1 if x == ">50K" else 0)

    X = dataset.drop("income", axis=1)
    y = dataset["income"]

    # One-hot Encoding
    categorical_cols = X.select_dtypes(include=["object"]).columns
    encoder = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore", categories=categories
    )
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
    X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    X = pd.concat([X.drop(categorical_cols, axis=1), X_encoded], axis=1)

    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, X_train.shape[1]


class IncomeClassifier(nn.Module):
    def __init__(self, input_dim):
        super(IncomeClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


def train(model, train_loader, num_epochs=5):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def evaluate(model, test_loader):
    model.eval()
    criterion = nn.BCELoss()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    loss = loss / len(test_loader)
    print(f"Accuracy: {accuracy:.2f}")
    return loss, accuracy


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_weights(net):
    ndarrays = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return ndarrays


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy}


# Initialize the model with dynamic input dimension from a sample partition
train_loader, test_loader, input_dim = load_data(partition_id=0)
net = IncomeClassifier(input_dim)
params = ndarrays_to_parameters(get_weights(net))

# Create Strategy
strategy = FedAvg(
    initial_parameters=params,
)

server = ServerApp(
    strategy=strategy,
    config=ServerConfig(num_rounds=3),
)


def client_fn(cid: str) -> Client:
    train_loader, test_loader, input_dim = load_data(partition_id=int(cid))
    net = IncomeClassifier(input_dim)
    return FlowerClient(net, train_loader, test_loader).to_client()


client = ClientApp(client_fn)

run_simulation(server_app=server, client_app=client, num_supernodes=5)
