from typing import List, Tuple
from flwr.server.server import Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.fedmd import FedMDStrategy
from flwr.proto import fedmd_pb2

# Distill RPC 호출 헬퍼 (gRPC/Driver 내부 API에 의존하지 않도록 간단화 시뮬)
# 실제 Flower 내부 Driver를 사용하려면 driver API를 통해 custom instruction을 전달해야 함.
# 여기서는 simulation 환경에서 직접 클라이언트 객체를 호출하는 방식으로 예제를 제공.
# (simulation에서 client_fn이 실제 객체를 반환하도록 구성)
def run_fedmd_round(
    server,
    strategy: FedMDStrategy,
    rnd: int,
    client_manager,
    clients: List,
    validator=None,
):
    print(f"\n🔄 Starting FedMD Round {rnd}")
    
    # 1) Distill 단계
    print(f"  📊 Collecting logits from {len(clients)} clients...")
    pairs = strategy.configure_distill(rnd, client_manager)
    distill_results: List[Tuple[ClientProxy, fedmd_pb2.DistillRes]] = []
    for c, ins in pairs:
        # simulation에서 ClientProxy가 실제 객체로 get_public_logits 호출 가능하도록 구성
        res = c.get_public_logits(ins.public_id, list(ins.sample_ids))
        distill_results.append((c, res))

    print(f"  🔄 Aggregating logits...")
    consensus = strategy.aggregate_logits(rnd, distill_results, failures=[])
    
    # 로짓 합의 분석
    if validator:
        from flwr.common.tensor import tensor_to_ndarray
        consensus_logits = tensor_to_ndarray(consensus.avg_logits)
        validator.analyze_logit_consensus(rnd, consensus_logits)

    # 2) Distill Fit 단계
    print(f"  🎯 Performing distillation training...")
    pairs_fit = strategy.configure_distill_fit(rnd, consensus, client_manager)
    for c, cons in pairs_fit:
        _ = c.distill_fit(cons, temperature=strategy.temperature, epochs=strategy.distill_epochs)
    
    # 라운드 후 모델 성능 평가
    if validator:
        validator.evaluate_client_models(rnd)
    
    print(f"✅ FedMD Round {rnd} completed")

def run_fedmd_training(server, strategy: FedMDStrategy, num_rounds: int, client_manager, clients: List, validator=None):
    for rnd in range(1, num_rounds + 1):
        run_fedmd_round(server, strategy, rnd, client_manager, clients, validator)
