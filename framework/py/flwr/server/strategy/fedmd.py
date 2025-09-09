from typing import List, Tuple, Dict, Any
import numpy as np

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitIns, FitRes

from flwr.proto import fedmd_pb2
from flwr.common.tensor import ndarray_to_tensor, tensor_to_ndarray

# Helper: 평균
def _average_logits(distill_results: List[fedmd_pb2.DistillRes]) -> fedmd_pb2.ConsensusIns:
    assert len(distill_results) > 0
    sample_ids = distill_results[0].sample_ids
    arrs = []
    for r in distill_results:
        # 동일한 sample_ids 전제
        arrs.append(tensor_to_ndarray(r.logits))
    avg = np.mean(arrs, axis=0)
    return fedmd_pb2.ConsensusIns(
        public_id=distill_results[0].public_id,
        sample_ids=sample_ids,
        avg_logits=ndarray_to_tensor(avg),
    )

class FedMDStrategy(Strategy):
    def __init__(
        self,
        public_id: str,
        public_sample_size: int = 2048,
        temperature: float = 1.0,
        distill_epochs: int = 1,
        batch_size: int = 64,
        public_sampler=None,  # callable: (public_id, rnd, n) -> List[int]
    ):
        self.public_id = public_id
        self.public_sample_size = public_sample_size
        self.temperature = temperature
        self.distill_epochs = distill_epochs
        self.batch_size = batch_size
        self.public_sampler = public_sampler

    # 표준 Strategy API (필요 최소한 구현)
    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return None

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # FedMD는 표준 fit 대신 "distill → distill_fit" 두 단계가 필요하므로
        # 여기서는 빈 FitIns를 반환하고, main loop에서 별도 오케스트레이션을 사용하도록 권고.
        return []

    def aggregate_fit(
        self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ) -> Tuple[Parameters | None, Dict[str, Any]]:
        return None, {}

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return []

    def aggregate_evaluate(
        self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ) -> Tuple[float | None, Dict[str, Any]]:
        return None, {}

    def evaluate(self, parameters: Parameters) -> Tuple[float, Dict[str, Any]]:
        return 0.0, {}

    # --- FedMD 전용 오케스트레이션 메서드 ---

    def configure_distill(self, rnd: int, client_manager: ClientManager) -> List[Tuple[ClientProxy, fedmd_pb2.DistillIns]]:
        sample_ids = self.public_sampler(self.public_id, rnd, self.public_sample_size) if self.public_sampler else list(range(self.public_sample_size))
        ins = fedmd_pb2.DistillIns(public_id=self.public_id, sample_ids=sample_ids)
        selected = client_manager.sample(num_clients=client_manager.num_available())  # 모두 선택(간단화)
        return [(c, ins) for c in selected]

    def aggregate_logits(
        self, rnd: int, results: List[Tuple[ClientProxy, fedmd_pb2.DistillRes]], failures: List[BaseException]
    ) -> fedmd_pb2.ConsensusIns:
        distill_res = [r for _, r in results]
        return _average_logits(distill_res)

    def configure_distill_fit(
        self, rnd: int, consensus: fedmd_pb2.ConsensusIns, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fedmd_pb2.ConsensusIns]]:
        selected = client_manager.sample(num_clients=client_manager.num_available())
        return [(c, consensus) for c in selected]
