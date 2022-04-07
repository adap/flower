Secure Aggregation Protocols
============================

Include SecAgg, SecAgg+, and LightSecAgg protocol. The LightSecAgg protocol has not been implemented yet, so its diagram and abstraction may not be accurate in practice.
The SecAgg protocol can be considered as a special case of the SecAgg+ protocol.

The :code:`SecAgg+` abstraction
-------------------------------

In this implementation, each client will be assigned with a unique index (int) for secure aggregation, and thus many python dictionaries used have keys of int type rather than ClientProxy type.

.. code-block:: python

    class SecAggPlusProtocol(ABC):
        """Abstract base class for the SecAgg+ protocol implementations."""

        @abstractmethod
        def generate_graph(
            self, clients: List[ClientProxy], k: int
        ) -> ClientGraph:
            """Build a k-degree undirected graph of clients.
            Each client will only generate pair-wise masks with its k neighbours.
            k is equal to the number of clients in SecAgg, i.e., a complete graph.
            This function may need extra inputs to decide on the generation of the graph."""

        @abstractmethod
        def setup_config(
            self, clients: List[ClientProxy], config_dict: Dict[str, Scalar]
        ) -> SetupConfigResultsAndFailures:
            """Configure the next round of secure aggregation. (SetupConfigRes is an empty class.)"""

        @abstractmethod
        def ask_keys(
            self,
            clients: List[ClientProxy], ask_keys_ins_list: List[AskKeysIns]
        ) -> AskKeysResultsAndFailures:
            """Ask public keys. (AskKeysIns is an empty class, and hence ask_keys_ins_list can be omitted.)"""

        @abstractmethod
        def share_keys(
            self,
            clients: List[ClientProxy], public_keys_dict: Dict[int, AskKeysRes],
            graph: ClientGraph
        ) -> ShareKeysResultsAndFailures:
            """Send public keys."""

        @abstractmethod
        def ask_vectors(
            clients: List[ClientProxy],
            forward_packet_list_dict: Dict[int, List[ShareKeysPacket]],
            client_instructions=None: Dict[int, FitIns]
        ) -> AskVectorsResultsAndFailures:
            """Ask vectors of local model parameters.
            (If client_instructions is not None, local models will be trained in the ask vectors stage,
            rather than trained parallelly as the protocol goes through the previous stages.)"""

        @abstractmethod
        def unmask_vectors(
            clients: List[ClientProxy],
            dropout_clients: List[ClientProxy],
            graph: ClientGraph
        ) -> UnmaskVectorsResultsAndFailures:
            """Unmask and compute the aggregated model. UnmaskVectorRes contains shares of keys needed to generate masks."""



The Flower server will execute and process received results in the following order:

.. mermaid::
  sequenceDiagram
      participant S as Flower Server
      participant P as SecAgg+ Protocol
      participant C1 as Flower Client
      participant C2 as Flower Client
      participant C3 as Flower Client

      S->>P: generate_graph
      activate P
      P-->>S: client_graph
      deactivate P

      Note left of P: Stage 0:<br/>Setup Config
      rect rgb(249, 219, 130)
      S->>P: setup_config<br/>clients, config_dict
      activate P
      P->>C1: SetupConfigIns
      deactivate P
      P->>C2:
      P->>C3:
      C1->>P: SetupConfigRes (empty)
      C2->>P:
      C3->>P:
      activate P
      P-->>S: None
      deactivate P
      end

      Note left of P: Stage 1:<br/>Ask Keys
      rect rgb(249, 219, 130)
      S->>P: ask_keys<br/>clients
      activate P
      P->>C1: AskKeysIns (empty)
      deactivate P
      P->>C2:
      P->>C3:
      C1->>P: AskKeysRes
      C2->>P:
      C3->>P:
      activate P
      P-->>S: public keys
      deactivate P
      end

      Note left of P: Stage 2:<br/>Share Keys
      rect rgb(249, 219, 130)
      S->>P: share_keys<br/>clients, public_keys_dict,<br/>client_graph
      activate P
      P->>C1: ShareKeysIns
      deactivate P
      P->>C2:
      P->>C3:
      C1->>P: ShareKeysRes
      C2->>P:
      C3->>P:
      activate P
      P-->>S: encryted key shares
      deactivate P
      end

      Note left of P: Stage 3:<br/>Ask Vectors
      rect rgb(249, 219, 130)
      S->>P: ask_vectors<br/>clients,<br/>forward_packet_list_dict
      activate P
      P->>C1: AskVectorsIns
      deactivate P
      P->>C2:
      P->>C3:
      C1->>P: AskVectorsRes
      C2->>P:
      activate P
      P-->>S: masked vectors
      deactivate P
      end

      Note left of P: Stage 4:<br/>Unmask Vectors
      rect rgb(249, 219, 130)
      S->>P: unmask_vectors<br/>clients, dropped_clients,<br/>client_graph
      activate P
      P->>C1: UnmaskVectorsIns
      deactivate P
      P->>C2:
      C1->>P: UnmaskVectorsRes
      C2->>P:
      activate P
      P-->>S: key shares
      deactivate P
      end

The :code:`LightSecAgg` abstraction
-----------------------------------

In this implementation, each client will be assigned with a unique index (int) for secure aggregation, and thus many python dictionaries used have keys of int type rather than ClientProxy type.

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
            clients: List[ClientProxy], public_keys_dict: Dict[int, LightSecAggSetupConfigRes]
        ) -> AskEncryptedEncodedMasksResultsAndFailures:
            """Ask encrypted encoded masks. The protocol adopts Diffie-Hellman keys to build pair-wise secured channels to transfer encoded mask."""

        @abstractmethod
        def ask_masked_models(
            self,
            clients: List[ClientProxy],
            forward_packet_list_dict: Dict[int, List[EncryptedEncodedMasksPacket]],
            client_instructions=None: Dict[int, FitIns]
        ) -> AskMaskedModelsResultsAndFailures:
            """Ask the masked local models.
            (If client_instructions is not None, local models will be trained in the ask vectors stage,
            rather than trained parallelly as the protocol goes through the previous stages.)"""

        @abstractmethod
        def ask_aggregated_encoded_masks(
            clients: List[ClientProxy]
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
      P->>C2:
      P->>C3:
      C1->>P: LightSecAggSetupConfigRes
      C2->>P:
      C3->>P:
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
      P->>C2:
      P->>C3:
      C1->>P: AskEncryptedEncodedMasksRes
      C2->>P:
      C3->>P:
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
      P->>C2:
      P->>C3:
      C1->>P: AskMaskedModelsRes
      C2->>P:
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
      P->>C2:
      C1->>P: AskAggregatedEncodedMasksRes
      C2->>P:
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

        FitResultsAndFailures = Tuple[
            List[Tuple[ClientProxy, FitRes]], List[BaseException]
        ]


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

