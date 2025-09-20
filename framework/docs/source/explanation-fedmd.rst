Federated Model Distillation (FedMD)
====================================

Federated Model Distillation (FedMD) is a federated learning approach that enables knowledge sharing through logit distillation on public data. Unlike traditional federated learning methods that share model parameters, FedMD allows clients to share their knowledge in the form of logits (model outputs) on a shared public dataset.

How FedMD Works
----------------

FedMD operates in three main phases:

1. **Distill Phase**: Each client generates logits (model outputs) on a shared public dataset
2. **Aggregation Phase**: The server collects and averages the logits from all clients to create a consensus
3. **Distillation Phase**: Each client trains their model using the consensus logits as soft targets

.. image:: ../_static/fedmd-workflow.png
   :alt: FedMD Workflow
   :align: center

Key Components
--------------

**Client-Side Components:**

- **FedMDNumPyClient**: Extends the standard NumPyClient with FedMD capabilities
- **PublicDataProvider**: Manages access to shared public datasets
- **Logit Generation**: Produces model outputs on public data samples
- **Distillation Training**: Learns from consensus logits using temperature-scaled soft targets

**Server-Side Components:**

- **FedMDStrategy**: Orchestrates the FedMD process
- **PublicDatasetRegistry**: Manages public dataset metadata and sampling
- **Logit Aggregation**: Averages client logits to create consensus
- **Consensus Distribution**: Sends aggregated logits back to clients

**Communication Protocol:**

- **Tensor Messages**: Efficient serialization of numpy arrays
- **DistillIns/DistillRes**: Request/response for logit collection
- **ConsensusIns**: Consensus logits for client training

Benefits of FedMD
-----------------

**Privacy-Preserving:**
- Only logits (not raw data or model parameters) are shared
- No direct access to private client data
- Maintains data privacy while enabling knowledge transfer

**Heterogeneous Model Support:**
- Different client architectures can participate
- No requirement for identical model structures
- Enables knowledge transfer between diverse models

**Effective Knowledge Transfer:**
- Soft targets provide richer information than hard labels
- Temperature scaling allows control over knowledge transfer intensity
- Consensus learning improves model generalization

**Scalable and Flexible:**
- Works with any number of clients
- Configurable sampling strategies for public data
- Adaptable to different domains and datasets

Use Cases
---------

**Heterogeneous Federated Learning:**
- Clients with different model architectures
- Varying computational capabilities
- Different data distributions

**Privacy-Sensitive Applications:**
- Medical data analysis
- Financial modeling
- Personal data processing

**Knowledge Transfer Scenarios:**
- Transfer learning in federated settings
- Model compression and distillation
- Cross-domain knowledge sharing

Example Usage
-------------

Here's a simple example of how to use FedMD:

.. code-block:: python

   from flwr.server.strategy import FedMDStrategy
   from flwr.client import FedMDNumPyClient
   from flwr.server.app_fedmd import run_fedmd_training

   # Create FedMD strategy
   strategy = FedMDStrategy(
       public_id="cifar10_v1",
       public_sample_size=512,
       temperature=2.0,
       distill_epochs=1
   )

   # Create clients with FedMD support
   clients = [FedMDNumPyClient(model, train_loader, public_provider) 
              for _ in range(num_clients)]

   # Run FedMD training
   run_fedmd_training(server, strategy, num_rounds=3, 
                     client_manager=client_manager, clients=clients)

For a complete working example, see :doc:`tutorial-fedmd-pytorch`.

Configuration Options
--------------------

**Strategy Parameters:**

- ``public_id``: Identifier for the public dataset
- ``public_sample_size``: Number of samples to use per round
- ``temperature``: Temperature for soft target scaling
- ``distill_epochs``: Number of distillation training epochs
- ``batch_size``: Batch size for distillation training
- ``public_sampler``: Custom sampling function for public data

**Client Parameters:**

- ``model``: The neural network model
- ``train_loader``: Private data loader (optional)
- ``public_provider``: Public data provider
- ``device``: Computing device (CPU/GPU)
- ``optimizer_ctor``: Optimizer constructor function

Best Practices
--------------

**Public Dataset Selection:**
- Choose representative public data
- Ensure sufficient diversity
- Consider domain relevance

**Temperature Tuning:**
- Higher temperatures (2.0-4.0) for more exploration
- Lower temperatures (1.0-2.0) for more exploitation
- Experiment with different values

**Sampling Strategy:**
- Use different samples per round
- Ensure fair representation
- Consider data augmentation

**Monitoring and Validation:**
- Track client-consensus convergence
- Monitor distillation loss
- Validate model performance

Limitations and Considerations
------------------------------

**Public Data Requirements:**
- Requires access to shared public data
- Data quality affects knowledge transfer
- May not be suitable for highly sensitive domains

**Communication Overhead:**
- Logits require more bandwidth than parameters
- Consider compression techniques
- Optimize for large-scale deployments

**Convergence Guarantees:**
- No theoretical convergence guarantees
- Performance depends on data similarity
- May require careful hyperparameter tuning

Related Work
------------

FedMD builds upon several key concepts:

- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"
- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks"
- **Model Distillation**: Buciluă et al., "Model Compression"

For more details on the implementation and advanced usage, see the API reference and tutorial sections.
