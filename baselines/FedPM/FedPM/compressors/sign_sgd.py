from flower_fl.compressors.compressor import Compressor

from typing import Dict
import torch
import math
import numpy as np
import random
from scipy.stats import entropy


class SignSGDCompressor(Compressor):
    def __init__(
            self,
            params: Dict,
            device: torch.device
    ) -> None:
        super().__init__(
            params=params,
            device=device
        )

    def compress(
            self,
            updates: torch.Tensor,
            compress_config: Dict = None,
            iter_num=None,
    ) -> torch.Tensor:
        compressed_delta = []
        num_params = 0
        num_ones = 0
        for i, (name, param) in enumerate(updates.items()):
            signed_param = torch.sign(param)
            compressed_delta.append(signed_param.cpu().numpy())
            # Add estimated ps
            num_param = torch.numel(signed_param)
            num_params += num_param
            num_ones += (torch.sum(signed_param) + num_param) / 2
        local_freq = num_ones / num_params
        local_bitrate = self.find_bitrate([local_freq + 1e-50, 1 - local_freq + 1e-50], num_params)

        return compressed_delta, (local_bitrate / num_params).cpu().numpy()


    def find_bitrate(self, probs, num_params):
        local_bitrate = 0
        for p in probs:
            local_bitrate += p * math.log2(1 / p)
        return local_bitrate * num_params


