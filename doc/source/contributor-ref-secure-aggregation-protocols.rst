Secure Aggregation Protocols
============================

Include SecAgg, SecAgg+, and LightSecAgg protocol. The LightSecAgg protocol has not been
implemented yet, so its diagram and abstraction may not be accurate in practice. The
SecAgg protocol can be considered as a special case of the SecAgg+ protocol.

The ``SecAgg+`` abstraction
---------------------------

In this implementation, each client will be assigned with a unique index (int) for
secure aggregation, and thus many python dictionaries used have keys of int type rather
than ClientProxy type.

The Flower server will execute and process received results in the following order:

.. mermaid::

    sequenceDiagram
        participant ServerApp as ServerApp (in SuperLink)
        participant SecAggPlusWorkflow
        participant ClientApp as secaggplus_mod
        participant RealClientApp as ClientApp (in SuperNode)

        ServerApp->>SecAggPlusWorkflow: invoke

        rect rgb(235, 235, 235)
        note over SecAggPlusWorkflow,ClientApp: Stage 0: Setup
        SecAggPlusWorkflow-->>ClientApp: Send SecAgg+ configuration
        ClientApp-->>SecAggPlusWorkflow: Send public keys
        end

        rect rgb(220, 220, 220)
        note over SecAggPlusWorkflow,ClientApp: Stage 1: Share Keys
        SecAggPlusWorkflow-->>ClientApp: Broadcast public keys
        ClientApp-->>SecAggPlusWorkflow: Send encrypted private key shares
        end

        rect rgb(235, 235, 235)
        note over SecAggPlusWorkflow,RealClientApp: Stage 2: Collect Masked Vectors
        SecAggPlusWorkflow-->>ClientApp: Forward the received shares
        ClientApp->>RealClientApp: fit instruction
        activate RealClientApp
        RealClientApp->>ClientApp: updated model
        deactivate RealClientApp
        ClientApp-->>SecAggPlusWorkflow: Send masked model parameters
        end

        rect rgb(220, 220, 220)
        note over SecAggPlusWorkflow,ClientApp: Stage 3: Unmask
        SecAggPlusWorkflow-->>ClientApp: Request private key shares
        ClientApp-->>SecAggPlusWorkflow: Send private key shares
        end
        SecAggPlusWorkflow->>SecAggPlusWorkflow: Unmask Aggregated Model
        SecAggPlusWorkflow->>ServerApp: Aggregated Model

The ``LightSecAgg`` abstraction
-------------------------------

In this implementation, each client will be assigned with a unique index (int) for
secure aggregation, and thus many python dictionaries used have keys of int type rather
than ClientProxy type.

.. code-block:: python

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
            clients: List[ClientProxy],
            public_keys_dict: Dict[int, LightSecAggSetupConfigRes],
        ) -> AskEncryptedEncodedMasksResultsAndFailures:
            """Ask encrypted encoded masks. The protocol adopts Diffie-Hellman keys to build pair-wise secured channels to transfer encoded mask."""

        @abstractmethod
        def ask_masked_models(
            self,
            clients: List[ClientProxy],
            forward_packet_list_dict: Dict[int, List[EncryptedEncodedMasksPacket]],
            client_instructions: Dict[int, FitIns] = None,
        ) -> AskMaskedModelsResultsAndFailures:
            """Ask the masked local models.
            (If client_instructions is not None, local models will be trained in the ask vectors stage,
            rather than trained parallelly as the protocol goes through the previous stages.)
            """

        @abstractmethod
        def ask_aggregated_encoded_masks(
            clients: List[ClientProxy],
        ) -> AskAggregatedEncodedMasksResultsAndFailures:
            """Ask aggregated encoded masks"""

The Flower server will execute and process received results in the following order:

.. mermaid::

    sequenceDiagram
        participant S as Flower Server
        participant P as LightSecAgg Protocol
        participant C1 as Flower Client
        participant C2 as Flower Client
        participant C3 as Flower Client

        Note left of P: Stage 0:<br/>Setup Config
        rect rgb(249, 219, 130)
        S->>P: setup_config<br/>clients, config_dict
        activate P
        P->>C1: LightSecAggSetupConfigIns
        deactivate P
        P->>C2: LightSecAggSetupConfigIns
        P->>C3: LightSecAggSetupConfigIns
        C1->>P: LightSecAggSetupConfigRes
        C2->>P: LightSecAggSetupConfigRes
        C3->>P: LightSecAggSetupConfigRes
        activate P
        P-->>S: public keys
        deactivate P
        end

        Note left of P: Stage 1:<br/>Ask Encrypted Encoded Masks
        rect rgb(249, 219, 130)
        S->>P: ask_encrypted_encoded_masks<br/>clients, public_keys_dict
        activate P
        P->>C1: AskEncryptedEncodedMasksIns
        deactivate P
        P->>C2: AskEncryptedEncodedMasksIns
        P->>C3: AskEncryptedEncodedMasksIns
        C1->>P: AskEncryptedEncodedMasksRes
        C2->>P: AskEncryptedEncodedMasksRes
        C3->>P: AskEncryptedEncodedMasksRes
        activate P
        P-->>S: forward packets
        deactivate P
        end

        Note left of P: Stage 2:<br/>Ask Masked Models
        rect rgb(249, 219, 130)
        S->>P: share_keys<br/>clients, forward_packet_list_dict
        activate P
        P->>C1: AskMaskedModelsIns
        deactivate P
        P->>C2: AskMaskedModelsIns
        P->>C3: AskMaskedModelsIns
        C1->>P: AskMaskedModelsRes
        C2->>P: AskMaskedModelsRes
        activate P
        P-->>S: masked local models
        deactivate P
        end

        Note left of P: Stage 3:<br/>Ask Aggregated Encoded Masks
        rect rgb(249, 219, 130)
        S->>P: ask_aggregated_encoded_masks<br/>clients
        activate P
        P->>C1: AskAggregatedEncodedMasksIns
        deactivate P
        P->>C2: AskAggregatedEncodedMasksIns
        C1->>P: AskAggregatedEncodedMasksRes
        C2->>P: AskAggregatedEncodedMasksRes
        activate P
        P-->>S: the aggregated model
        deactivate P
        end

Types
-----

.. code-block:: python

    # the SecAgg+ protocol

    ClientGraph = Dict[int, List[int]]

    SetupConfigResultsAndFailures = Tuple[
        List[Tuple[ClientProxy, SetupConfigRes]], List[BaseException]
    ]

    AskKeysResultsAndFailures = Tuple[
        List[Tuple[ClientProxy, AskKeysRes]], List[BaseException]
    ]

    ShareKeysResultsAndFailures = Tuple[
        List[Tuple[ClientProxy, ShareKeysRes]], List[BaseException]
    ]

    AskVectorsResultsAndFailures = Tuple[
        List[Tuple[ClientProxy, AskVectorsRes]], List[BaseException]
    ]

    UnmaskVectorsResultsAndFailures = Tuple[
        List[Tuple[ClientProxy, UnmaskVectorsRes]], List[BaseException]
    ]

    FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]


    @dataclass
    class SetupConfigIns:
        sec_agg_cfg_dict: Dict[str, Scalar]


    @dataclass
    class SetupConfigRes:
        pass


    @dataclass
    class AskKeysIns:
        pass


    @dataclass
    class AskKeysRes:
        """Ask Keys Stage Response from client to server"""

        pk1: bytes
        pk2: bytes


    @dataclass
    class ShareKeysIns:
        public_keys_dict: Dict[int, AskKeysRes]


    @dataclass
    class ShareKeysPacket:
        source: int
        destination: int
        ciphertext: bytes


    @dataclass
    class ShareKeysRes:
        share_keys_res_list: List[ShareKeysPacket]


    @dataclass
    class AskVectorsIns:
        ask_vectors_in_list: List[ShareKeysPacket]
        fit_ins: FitIns


    @dataclass
    class AskVectorsRes:
        parameters: Parameters


    @dataclass
    class UnmaskVectorsIns:
        available_clients: List[int]
        dropout_clients: List[int]


    @dataclass
    class UnmaskVectorsRes:
        share_dict: Dict[int, bytes]


    # the LightSecAgg protocol

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


    @dataclass
    class LightSecAggSetupConfigIns:
        sec_agg_cfg_dict: Dict[str, Scalar]


    @dataclass
    class LightSecAggSetupConfigRes:
        pk: bytes


    @dataclass
    class AskEncryptedEncodedMasksIns:
        public_keys_dict: Dict[int, LightSecAggSetupConfigRes]


    @dataclass
    class EncryptedEncodedMasksPacket:
        source: int
        destination: int
        ciphertext: bytes


    @dataclass
    class AskEncryptedEncodedMasksRes:
        packet_list: List[EncryptedEncodedMasksPacket]


    @dataclass
    class AskMaskedModelsIns:
        packet_list: List[EncryptedEncodedMasksPacket]
        fit_ins: FitIns


    @dataclass
    class AskMaskedModelsRes:
        parameters: Parameters


    @dataclass
    class AskAggregatedEncodedMasksIns:
        surviving_clients: List[int]


    @dataclass
    class AskAggregatedEncodedMasksRes:
        aggregated_encoded_mask: Parameters
