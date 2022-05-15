from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from flwr.common.typing import Scalar, LightSecAggSetupConfigRes, AskEncryptedEncodedMasksRes, AskMaskedModelsRes, \
    AskAggregatedEncodedMasksRes, FitIns, EncryptedEncodedMasksPacket, Parameters, FitRes
from flwr.server.client_proxy import ClientProxy

LightSecAggSetupConfigResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, LightSecAggSetupConfigRes]], List[BaseException]
]

AskEncryptedEncodedMasksResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, AskEncryptedEncodedMasksRes]], List[BaseException]
]

AskMaskedModelsResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, AskMaskedModelsRes]], List[BaseException]
]

AskAggregatedEncodedMasksResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, AskAggregatedEncodedMasksRes]], List[BaseException]
]

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]


class LightSecAggProtocol(ABC):
    """Abstract base class for the LightSecAgg protocol implementations."""

    @abstractmethod
    def setup_config(
        self, clients: List[ClientProxy], config_dict: Dict[str, Scalar]
    ) -> LightSecAggSetupConfigResultsAndFailures:
        """Configure the next round of secure aggregation."""

    @abstractmethod
    def ask_encrypted_encoded_masks(
        self,
        clients: List[ClientProxy], public_keys_dict: Dict[int, LightSecAggSetupConfigRes]
    ) -> AskEncryptedEncodedMasksResultsAndFailures:
        """Ask encrypted encoded masks. The protocol adopts Diffie-Hellman keys to build pair-wise secured channels to transfer encoded mask."""

    @abstractmethod
    def ask_masked_models(
        self,
        clients: List[ClientProxy],
        forward_packet_list_dict: Dict[int, List[EncryptedEncodedMasksPacket]],
        client_instructions: Dict[int, FitIns] = None
    ) -> AskMaskedModelsResultsAndFailures:
        """Ask the masked local models.
        (If client_instructions is not None, local models will be trained in the ask vectors stage,
        rather than trained parallelly as the protocol goes through the previous stages.)"""

    @abstractmethod
    def ask_aggregated_encoded_masks(
        self,
        clients: List[ClientProxy]
    ) -> AskAggregatedEncodedMasksResultsAndFailures:
        """Ask aggregated encoded masks"""


class SecureAggregationFitRound(ABC):
    @abstractmethod
    def fit_round(self, server, rnd: int) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        """fit round"""
