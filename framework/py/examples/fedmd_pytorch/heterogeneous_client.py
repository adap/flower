import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

from examples.fedmd_pytorch.heterogeneous_models import get_model_for_client, print_model_info
from flwr.client.fedmd_numpy_client import FedMDNumPyClient
from flwr.client.public_data.provider import PublicDataProvider

class LocalClientProxy:
    """Simulation용 간단 프록시: 서버가 직접 .get_public_logits/.distill_fit 호출"""
    def __init__(self, client: FedMDNumPyClient, client_id: int):
        self.client = client
        self.client_id = client_id

    def get_public_logits(self, public_id, sample_ids):
        return self.client.get_public_logits(public_id, sample_ids)

    def distill_fit(self, consensus, temperature, epochs):
        return self.client.distill_fit(consensus, temperature=temperature, epochs=epochs)

def make_heterogeneous_client(client_id: int, device="cpu"):
    """이기종 모델을 사용하는 클라이언트 생성"""
    tfm = T.Compose([T.ToTensor()])
    
    # Private data (각 클라이언트마다 다른 데이터 분할)
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    n = len(train_ds)
    
    # 클라이언트별로 다른 데이터 분할 (이기종 데이터 분포 시뮬레이션)
    if client_id == 0:
        # 클라이언트 1: 처음 50% 데이터
        train_sub, _ = random_split(train_ds, [int(0.5*n), n - int(0.5*n)])
    elif client_id == 1:
        # 클라이언트 2: 중간 50% 데이터
        start_idx = int(0.25*n)
        end_idx = int(0.75*n)
        indices = list(range(start_idx, end_idx))
        train_sub = torch.utils.data.Subset(train_ds, indices)
    else:
        # 클라이언트 3: 마지막 50% 데이터
        train_sub, _ = random_split(train_ds, [n - int(0.5*n), int(0.5*n)])
    
    train_loader = DataLoader(train_sub, batch_size=64, shuffle=True)

    # Public data (모든 클라이언트가 동일하게 접근)
    public_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    public_provider = PublicDataProvider(public_ds)

    # 클라이언트별로 다른 모델 사용
    model = get_model_for_client(client_id, num_classes=10)
    model = model.to(device)
    
    # 모델 정보 출력
    print_model_info(model, client_id)
    
    # 클라이언트별로 다른 옵티마이저 사용
    if client_id == 0:
        # 작은 모델: 높은 학습률
        optimizer_ctor = lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
    elif client_id == 1:
        # 중간 모델: 중간 학습률
        optimizer_ctor = lambda params: torch.optim.Adam(params, lr=0.001)
    else:
        # 큰 모델: 낮은 학습률
        optimizer_ctor = lambda params: torch.optim.AdamW(params, lr=0.0005, weight_decay=1e-4)
    
    client = FedMDNumPyClient(
        model=model, 
        train_loader=train_loader, 
        public_provider=public_provider, 
        device=device,
        optimizer_ctor=optimizer_ctor
    )
    
    return LocalClientProxy(client, client_id)
