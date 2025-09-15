import torchvision
import torchvision.transforms as T
from flwr.common.public_manifest import PublicManifest

def get_public_dataset(train=True):
    tfm = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=tfm)
    return ds

def get_manifest(ds) -> PublicManifest:
    return PublicManifest(public_id="cifar10_v1", num_samples=len(ds), classes=[str(i) for i in range(10)])
