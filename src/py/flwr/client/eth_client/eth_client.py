# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower client (abstract base class)."""
import time
from abc import ABC
from typing import Dict

import numpy as np
import torch.nn
from web3.exceptions import ContractLogicError

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
)
from .eth_base_client import _EthClient
from .ipfs_client import IPFSClient


class EthClient(ABC):
    """Abstract base class for Flower clients."""
    CONTRACT_ADDRESS = "0xF16165f1046f1B3cDB37dA25E835B986E696313A"
    ROUND_DURATION = 20000
    def __init__(self,
                 cid: str,
                 ):
        self.cid = cid
        self.EthBase = _EthClient(cid, self.CONTRACT_ADDRESS, deploy=True)
        self.IPFSClient = IPFSClient("/ip4/127.0.0.1/tcp/5001")
        self.round = None

    def initial_setting(self):
        while True:
            try:
                round = self.EthBase.currentRound()
                break
            except ContractLogicError as e:
                if int(self.cid) == 0:
                    self.set_genesis()
                    self.set_modelArchitecture()
            else:
                time.sleep(1)
    def set_genesis(self):
            genesis_cid = self.IPFSClient.add_model(self.IPFSClient.model)
            print(f"Genesis model cid : {genesis_cid}")
            tx = self.EthBase.setGenesis(genesis_cid)
            print("wait_for_tx")
            self.EthBase.wait_for_tx(tx)
            print("tx done")

    def set_modelArchitecture(self):
            arch_cid = self.IPFSClient.architecture_to_ipfs(self.IPFSClient.model)
            print(f"arch cid : {arch_cid}")
            tx = self.EthBase.setModelArch(arch_cid)
            print("wait_for_tx")
            self.EthBase.wait_for_tx(tx)
            print("tx done")

    def skip_round(self):
            round = self.EthBase.currentRound()
            tx = self.EthBase.skipRound(round)
            self.EthBase.wait_for_tx(tx)

    def module_to_properties(self,module: torch.nn.Module) -> Dict[str, np.ndarray]:
        properties = {}
        for name, param in module.named_parameters():
            properties[name] = param.detach().numpy()
        return properties


    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Return set of client's properties.

        Parameters
        ----------
        ins : GetPropertiesIns
            The get properties instructions received from the server containing
            a dictionary of configuration values.

        Returns
        -------
        GetPropertiesRes
            The current client properties.
        """

        net = self.IPFSClient.get_model(model_cid=None)
        properties = self.module_to_properties(net)

        return properties

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return the current local model parameters.

        Parameters
        ----------
        ins : GetParametersIns
            The get parameters instructions received from the server containing
            a dictionary of configuration values.

        Returns
        -------
        GetParametersRes
            The current local model parameters.
        """

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided parameters using the locally held dataset.

        Parameters
        ----------
        ins : FitIns
            The training instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local training process.

        Returns
        -------
        FitRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        ins : EvaluateIns
            The evaluation instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local evaluation process.

        Returns
        -------
        EvaluateRes
            The evaluation result containing the loss on the local dataset and
            other details such as the number of local data examples used for
            evaluation.
        """


def has_get_properties(client: EthClient) -> bool:
    """Check if NumPyClient implements get_properties."""
    return type(client).get_properties != EthClient.get_properties


def has_get_parameters(client: EthClient) -> bool:
    """Check if NumPyClient implements get_parameters."""
    return type(client).get_parameters != EthClient.get_parameters


def has_fit(client: EthClient) -> bool:
    """Check if NumPyClient implements fit."""
    return type(client).fit != EthClient.fit


def has_evaluate(client: EthClient) -> bool:
    """Check if NumPyClient implements evaluate."""
    return type(client).evaluate != EthClient.evaluate

