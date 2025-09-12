from flwr.server.strategy.fedmd import FedMDStrategy
from flwr.server.public_data.registry import PublicDatasetRegistry
from flwr.server.app_fedmd import run_fedmd_training
from examples.fedmd_pytorch.public_data import get_public_dataset, get_manifest

def build_server_and_strategy():
    # 퍼블릭 레지스트리
    ds = get_public_dataset(train=False)
    manifest = get_manifest(ds)

    registry = PublicDatasetRegistry()
    registry.register(manifest)

    def sampler(public_id: str, rnd: int, n: int):
        # 간단 샘플러 (라운드/시드 기반)
        import random
        random.seed(1000 + rnd)
        idx = list(range(manifest.num_samples))
        random.shuffle(idx)
        return idx[:n]

    strategy = FedMDStrategy(
        public_id=manifest.public_id,
        public_sample_size=512,
        temperature=2.0,
        distill_epochs=1,
        batch_size=64,
        public_sampler=sampler,
    )
    return strategy
