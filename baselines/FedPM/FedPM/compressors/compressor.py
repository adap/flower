from typing import Dict
import torch


class Compressor:
    def __init__(
            self,
            params: Dict,
            device: torch.device
    ) -> None:
        self.params = params
        self.device = device

    def compress(
            self,
            posterior_update: torch.Tensor,
            prior=None,
            compress_config: Dict = None,
            old_ids=None
    ):
        return posterior_update
