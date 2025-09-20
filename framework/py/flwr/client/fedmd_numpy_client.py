from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np

from flwr.client import NumPyClient
from flwr.client.public_data.provider import PublicDataProvider
from flwr.common import FitRes, Parameters
from flwr.common.typing import NDArrays

from flwr.proto import fedmd_pb2
from flwr.common.tensor import ndarray_to_tensor, tensor_to_ndarray

def _to_device(x, device):
    return x.to(device) if torch.is_tensor(x) else torch.tensor(x, device=device)

class FedMDNumPyClient(NumPyClient):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,  # private data loader (for optional local training)
        public_provider: PublicDataProvider,
        device: str = "cpu",
        optimizer_ctor=lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.public = public_provider
        self.device = device
        self.optimizer = optimizer_ctor(self.model.parameters())

    # 표준 NumPyClient 메서드들 (파라미터 관리)
    def get_parameters(self, config: Dict[str, str] = None) -> NDArrays:
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, str]) -> FitRes:
        # 기존 FedAvg 등과 호환을 위해 남겨둠(옵션). FedMD는 distill_fit 사용
        if parameters is not None:
            self.set_parameters(parameters)
        # no-op or light local training on private data if desired
        return FitRes(parameters=self.get_parameters(), num_examples=0, metrics={}, status={})

    def evaluate(self, parameters: NDArrays, config: Dict[str, str]):
        if parameters is not None:
            self.set_parameters(parameters)
        self.model.eval()
        # Dummy 0-eval to keep interface simple (user can implement)
        return 0.0, 0, {"acc": 0.0}

    # --- FedMD 확장 API (서버가 직접 호출하는 형태는 Strategy에서 처리) ---

    def get_public_logits(self, public_id: str, sample_ids: List[int]) -> fedmd_pb2.DistillRes:
        self.model.eval()
        with torch.no_grad():
            x = self.public.get_samples(sample_ids).to(self.device)
            logits = self.model(x)  # [N, num_classes]
            arr = logits.detach().cpu().numpy()
        return fedmd_pb2.DistillRes(
            public_id=public_id,
            sample_ids=sample_ids,
            logits=ndarray_to_tensor(arr),
        )

    def distill_fit(
        self,
        consensus: fedmd_pb2.ConsensusIns,
        temperature: float = 1.0,
        epochs: int = 1,
        batch_size: int = 64,
    ) -> FitRes:
        self.model.train()
        sample_ids = list(consensus.sample_ids)
        avg_logits = tensor_to_ndarray(consensus.avg_logits)  # [N, C]

        x_all = self.public.get_samples(sample_ids).to(self.device)
        y_soft = torch.tensor(avg_logits, dtype=torch.float32, device=self.device)
        # soft targets
        y_soft = F.softmax(y_soft / temperature, dim=-1)

        N = x_all.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                xb = x_all[idx]
                yb = y_soft[idx]

                self.optimizer.zero_grad()
                out = self.model(xb) / temperature
                logp = F.log_softmax(out, dim=-1)
                loss = F.kl_div(logp, yb, reduction="batchmean") * (temperature**2)
                loss.backward()
                self.optimizer.step()

        return FitRes(parameters=self.get_parameters(), num_examples=N, metrics={"distill_epochs": epochs}, status={})
