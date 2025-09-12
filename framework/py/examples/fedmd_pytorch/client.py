import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

from examples.fedmd_pytorch.model import SmallCifarNet
from flwr.client.fedmd_numpy_client import FedMDNumPyClient
from flwr.client.public_data.provider import PublicDataProvider

class LocalClientProxy:
    """Simulation용 간단 프록시: 서버가 직접 .get_public_logits/.distill_fit 호출"""
    def __init__(self, client: FedMDNumPyClient):
        self.client = client

    def get_public_logits(self, public_id, sample_ids):
        return self.client.get_public_logits(public_id, sample_ids)

    def distill_fit(self, consensus, temperature, epochs):
        return self.client.distill_fit(consensus, temperature=temperature, epochs=epochs)

def make_client(device="cpu"):
    tfm = T.Compose([T.ToTensor()])
    # private(train) 데이터 (임의 분할)
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    n = len(train_ds)
    train_sub, _ = random_split(train_ds, [int(0.5*n), n - int(0.5*n)])
    train_loader = DataLoader(train_sub, batch_size=64, shuffle=True)

    # public(test) 데이터
    public_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    public_provider = PublicDataProvider(public_ds)

    model = SmallCifarNet(num_classes=10)
    client = FedMDNumPyClient(model=model, train_loader=train_loader, public_provider=public_provider, device=device)
    return LocalClientProxy(client)
