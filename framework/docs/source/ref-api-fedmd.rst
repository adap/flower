FedMD API Reference
===================

This page provides detailed API documentation for the Federated Model Distillation (FedMD) components in Flower.

Client API
----------

FedMDNumPyClient
~~~~~~~~~~~~~~~~

.. autoclass:: flwr.client.fedmd_numpy_client.FedMDNumPyClient
   :members:
   :undoc-members:
   :show-inheritance:

**Constructor Parameters:**

.. py:parameter:: model: torch.nn.Module
   The neural network model to be trained

.. py:parameter:: train_loader: DataLoader
   Private data loader for optional local training

.. py:parameter:: public_provider: PublicDataProvider
   Provider for accessing public dataset

.. py:parameter:: device: str
   Computing device (default: "cpu")

.. py:parameter:: optimizer_ctor: callable
   Optimizer constructor function (default: SGD with lr=0.01, momentum=0.9)

**Methods:**

.. py:method:: get_public_logits(public_id: str, sample_ids: List[int]) -> DistillRes
   Generate logits for specified public data samples

   :param public_id: Identifier for the public dataset
   :param sample_ids: List of sample indices to process
   :return: DistillRes containing logits and metadata

.. py:method:: distill_fit(consensus: ConsensusIns, temperature: float = 1.0, epochs: int = 1, batch_size: int = 64) -> FitRes
   Perform distillation training using consensus logits

   :param consensus: Consensus logits from server
   :param temperature: Temperature for soft target scaling
   :param epochs: Number of distillation epochs
   :param batch_size: Batch size for training
   :return: FitRes with updated parameters and metrics

PublicDataProvider
~~~~~~~~~~~~~~~~~~

.. autoclass:: flwr.client.public_data.provider.PublicDataProvider
   :members:
   :undoc-members:
   :show-inheritance:

**Constructor Parameters:**

.. py:parameter:: dataset: Dataset
   PyTorch dataset containing public data

**Methods:**

.. py:method:: get_samples(sample_ids: List[int]) -> torch.Tensor
   Retrieve samples by their indices

   :param sample_ids: List of sample indices
   :return: Tensor containing the requested samples

Server API
----------

FedMDStrategy
~~~~~~~~~~~~~

.. autoclass:: flwr.server.strategy.fedmd.FedMDStrategy
   :members:
   :undoc-members:
   :show-inheritance:

**Constructor Parameters:**

.. py:parameter:: public_id: str
   Identifier for the public dataset

.. py:parameter:: public_sample_size: int
   Number of public samples to use per round (default: 2048)

.. py:parameter:: temperature: float
   Temperature for soft target scaling (default: 1.0)

.. py:parameter:: distill_epochs: int
   Number of distillation epochs per round (default: 1)

.. py:parameter:: batch_size: int
   Batch size for distillation training (default: 64)

.. py:parameter:: public_sampler: callable
   Custom sampling function for public data (default: None)

**Methods:**

.. py:method:: configure_distill(rnd: int, client_manager: ClientManager) -> List[Tuple[ClientProxy, DistillIns]]
   Configure the distill phase for a round

   :param rnd: Current round number
   :param client_manager: Client manager instance
   :return: List of client-proxy and instruction pairs

.. py:method:: aggregate_logits(rnd: int, results: List[Tuple[ClientProxy, DistillRes]], failures: List[BaseException]) -> ConsensusIns
   Aggregate logits from clients to create consensus

   :param rnd: Current round number
   :param results: List of client results
   :param failures: List of failures
   :return: ConsensusIns containing averaged logits

.. py:method:: configure_distill_fit(rnd: int, consensus: ConsensusIns, client_manager: ClientManager) -> List[Tuple[ClientProxy, ConsensusIns]]
   Configure the distillation training phase

   :param rnd: Current round number
   :param consensus: Consensus logits
   :param client_manager: Client manager instance
   :return: List of client-proxy and consensus pairs

PublicDatasetRegistry
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: flwr.server.public_data.registry.PublicDatasetRegistry
   :members:
   :undoc-members:
   :show-inheritance:

**Methods:**

.. py:method:: register(manifest: PublicManifest) -> None
   Register a public dataset manifest

   :param manifest: PublicManifest containing dataset metadata

.. py:method:: sample(public_id: str, n: int, seed: int = 0) -> List[int]
   Sample indices from a registered dataset

   :param public_id: Dataset identifier
   :param n: Number of samples to return
   :param seed: Random seed for reproducibility
   :return: List of sample indices

Common API
----------

PublicManifest
~~~~~~~~~~~~~~

.. autoclass:: flwr.common.public_manifest.PublicManifest
   :members:
   :undoc-members:
   :show-inheritance:

**Attributes:**

.. py:attribute:: public_id: str
   Unique identifier for the public dataset

.. py:attribute:: num_samples: int
   Total number of samples in the dataset

.. py:attribute:: hashes: Optional[List[str]]
   Optional integrity hashes for data validation

.. py:attribute:: classes: Optional[List[str]]
   Optional class labels for the dataset

Tensor Utilities
~~~~~~~~~~~~~~~~

.. autofunction:: flwr.common.tensor.ndarray_to_tensor
   :noindex:

.. autofunction:: flwr.common.tensor.tensor_to_ndarray
   :noindex:

**Functions:**

.. py:function:: ndarray_to_tensor(arr: np.ndarray) -> Tensor
   Convert numpy array to protobuf Tensor

   :param arr: Numpy array to convert
   :return: Protobuf Tensor message

.. py:function:: tensor_to_ndarray(t: Tensor) -> np.ndarray
   Convert protobuf Tensor to numpy array

   :param t: Protobuf Tensor message
   :return: Numpy array

Protocol Messages
-----------------

Tensor
~~~~~~

.. autoclass:: flwr.proto.fedmd_pb2.Tensor
   :members:
   :undoc-members:
   :show-inheritance:

**Fields:**

.. py:attribute:: shape: repeated int64
   Shape of the tensor

.. py:attribute:: buffer: bytes
   Raw tensor data as bytes

.. py:attribute:: dtype: string
   Data type of the tensor

DistillIns
~~~~~~~~~~

.. autoclass:: flwr.proto.fedmd_pb2.DistillIns
   :members:
   :undoc-members:
   :show-inheritance:

**Fields:**

.. py:attribute:: public_id: string
   Identifier for the public dataset

.. py:attribute:: sample_ids: repeated int64
   List of sample indices to process

DistillRes
~~~~~~~~~~

.. autoclass:: flwr.proto.fedmd_pb2.DistillRes
   :members:
   :undoc-members:
   :show-inheritance:

**Fields:**

.. py:attribute:: public_id: string
   Identifier for the public dataset

.. py:attribute:: sample_ids: repeated int64
   List of processed sample indices

.. py:attribute:: logits: Tensor
   Model logits for the samples

ConsensusIns
~~~~~~~~~~~~

.. autoclass:: flwr.proto.fedmd_pb2.ConsensusIns
   :members:
   :undoc-members:
   :show-inheritance:

**Fields:**

.. py:attribute:: public_id: string
   Identifier for the public dataset

.. py:attribute:: sample_ids: repeated int64
   List of sample indices

.. py:attribute:: avg_logits: Tensor
   Averaged logits from all clients

Training Utilities
------------------

run_fedmd_training
~~~~~~~~~~~~~~~~~~

.. autofunction:: flwr.server.app_fedmd.run_fedmd_training
   :noindex:

**Function:**

.. py:function:: run_fedmd_training(server, strategy: FedMDStrategy, num_rounds: int, client_manager, clients: List, validator=None) -> None
   Run complete FedMD training process

   :param server: Server instance (can be None for simulation)
   :param strategy: FedMDStrategy instance
   :param num_rounds: Number of training rounds
   :param client_manager: Client manager instance
   :param clients: List of client instances
   :param validator: Optional validator for monitoring

run_fedmd_round
~~~~~~~~~~~~~~~

.. autofunction:: flwr.server.app_fedmd.run_fedmd_round
   :noindex:

**Function:**

.. py:function:: run_fedmd_round(server, strategy: FedMDStrategy, rnd: int, client_manager, clients: List, validator=None) -> None
   Run a single FedMD round

   :param server: Server instance (can be None for simulation)
   :param strategy: FedMDStrategy instance
   :param rnd: Round number
   :param client_manager: Client manager instance
   :param clients: List of client instances
   :param validator: Optional validator for monitoring

Data Types
----------

DTYPE_TO_STR
~~~~~~~~~~~~

.. py:data:: DTYPE_TO_STR
   :type: Dict[np.dtype, str]

   Mapping from numpy dtypes to string representations

STR_TO_DTYPE
~~~~~~~~~~~~

.. py:data:: STR_TO_DTYPE
   :type: Dict[str, np.dtype]

   Mapping from string representations to numpy dtypes

Example Usage
-------------

Basic FedMD Setup
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from flwr.server.strategy import FedMDStrategy
   from flwr.client import FedMDNumPyClient
   from flwr.server.app_fedmd import run_fedmd_training

   # Create strategy
   strategy = FedMDStrategy(
       public_id="cifar10_v1",
       public_sample_size=512,
       temperature=2.0,
       distill_epochs=1
   )

   # Create clients
   clients = [FedMDNumPyClient(model, train_loader, public_provider) 
              for _ in range(num_clients)]

   # Run training
   run_fedmd_training(server, strategy, num_rounds=3, 
                     client_manager=client_manager, clients=clients)

Custom Sampling Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def custom_sampler(public_id: str, rnd: int, n: int) -> List[int]:
       # Custom sampling logic
       import random
       random.seed(1000 + rnd)
       indices = list(range(10000))  # Assuming 10k samples
       random.shuffle(indices)
       return indices[:n]

   strategy = FedMDStrategy(
       public_id="custom_dataset",
       public_sampler=custom_sampler,
       # ... other parameters
   )

Error Handling
--------------

Common exceptions and how to handle them:

**ImportError**: Missing dependencies
   - Ensure all required packages are installed
   - Check protobuf version compatibility

**ValueError**: Invalid parameters
   - Verify public_id exists in registry
   - Check sample_ids are within valid range

**RuntimeError**: Model or data issues
   - Ensure model is properly initialized
   - Verify public data is accessible

**MemoryError**: Insufficient memory
   - Reduce batch_size or public_sample_size
   - Use smaller models or datasets

For more examples and advanced usage, see the tutorial sections.
