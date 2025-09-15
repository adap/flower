from typing import Dict, List
from flwr.common.public_manifest import PublicManifest
import random

class PublicDatasetRegistry:
    def __init__(self) -> None:
        self._manifests: Dict[str, PublicManifest] = {}

    def register(self, manifest: PublicManifest) -> None:
        self._manifests[manifest.public_id] = manifest

    def sample(self, public_id: str, n: int, seed: int = 0) -> List[int]:
        mf = self._manifests[public_id]
        rng = random.Random(seed)
        idx = list(range(mf.num_samples))
        rng.shuffle(idx)
        return idx[:n]
