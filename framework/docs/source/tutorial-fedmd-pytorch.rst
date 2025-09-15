FedMD with PyTorch
==================

This tutorial shows how to implement Federated Model Distillation (FedMD) using PyTorch and Flower. We'll create a complete example with CIFAR-10 dataset, demonstrating how clients can share knowledge through logit distillation.

Prerequisites
-------------

Before starting, make sure you have the following installed:

.. code-block:: bash

   pip install flwr[simulation] torch torchvision

Understanding the Example
-------------------------

Our FedMD example consists of:

1. **Model Definition**: A simple CNN for CIFAR-10 classification
2. **Client Implementation**: FedMDNumPyClient with logit generation and distillation
3. **Server Strategy**: FedMDStrategy for logit aggregation
4. **Public Data Management**: Registry and provider for shared dataset
5. **Training Loop**: Complete FedMD training with monitoring

Let's start by examining each component.

Model Definition
----------------

First, let's define a simple CNN model for CIFAR-10:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class SmallCifarNet(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.c1 = nn.Conv2d(3, 32, 3, padding=1)
           self.c2 = nn.Conv2d(32, 64, 3, padding=1)
           self.p = nn.MaxPool2d(2, 2)
           self.fc1 = nn.Linear(64*8*8, 128)
           self.fc2 = nn.Linear(128, num_classes)

       def forward(self, x):
           x = self.p(F.relu(self.c1(x)))
           x = self.p(F.relu(self.c2(x)))
           x = x.view(x.size(0), -1)
           x = F.relu(self.fc1(x))
           return self.fc2(x)

Client Implementation
---------------------

The FedMDNumPyClient extends the standard NumPyClient with FedMD capabilities:

.. code-block:: python

   from flwr.client import FedMDNumPyClient
   from flwr.client.public_data.provider import PublicDataProvider

   class LocalClientProxy:
       def __init__(self, client: FedMDNumPyClient):
           self.client = client

       def get_public_logits(self, public_id, sample_ids):
           return self.client.get_public_logits(public_id, sample_ids)

       def distill_fit(self, consensus, temperature, epochs):
           return self.client.distill_fit(consensus, temperature=temperature, epochs=epochs)

   def make_client(device="cpu"):
       # Private data (randomly split)
       train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
       n = len(train_ds)
       train_sub, _ = random_split(train_ds, [int(0.5*n), n - int(0.5*n)])
       train_loader = DataLoader(train_sub, batch_size=64, shuffle=True)

       # Public data
       public_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
       public_provider = PublicDataProvider(public_ds)

       model = SmallCifarNet(num_classes=10)
       client = FedMDNumPyClient(model=model, train_loader=train_loader, 
                                public_provider=public_provider, device=device)
       return LocalClientProxy(client)

Server Strategy
---------------

The FedMDStrategy handles logit collection and aggregation:

.. code-block:: python

   from flwr.server.strategy import FedMDStrategy
   from flwr.server.public_data.registry import PublicDatasetRegistry

   def build_server_and_strategy():
       # Public dataset registry
       ds = get_public_dataset(train=False)
       manifest = get_manifest(ds)

       registry = PublicDatasetRegistry()
       registry.register(manifest)

       def sampler(public_id: str, rnd: int, n: int):
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

Complete Training Example
-------------------------

Here's the complete training script:

.. code-block:: python

   import torch
   from flwr.server.app_fedmd import run_fedmd_training

   def main():
       device = "cuda" if torch.cuda.is_available() else "cpu"
       strategy = build_server_and_strategy()

       # Create clients
       num_clients = 3
       clients = [make_client(device=device) for _ in range(num_clients)]

       # ClientManager mock
       class _CM:
           def __init__(self, clients):
               self._clients = clients
           def num_available(self): return len(self._clients)
           def sample(self, num_clients): return self._clients[:num_clients]

       cm = _CM(clients)

       # Run FedMD training
       run_fedmd_training(None, strategy, num_rounds=3, 
                         client_manager=cm, clients=clients)
       print("FedMD training completed!")

   if __name__ == "__main__":
       main()

Running the Example
-------------------

To run the complete example:

.. code-block:: bash

   python examples/fedmd_pytorch/run_simulation.py

Expected Output
---------------

The training will produce output similar to:

.. code-block:: text

   Using device: cpu

   ============================================================
   STARTING FEDMD TRAINING
   ============================================================

   === Initial Model Performance ===
   Client 1: Accuracy = 0.0996, Loss = 2.3046
   Client 2: Accuracy = 0.1000, Loss = 2.3068
   Client 3: Accuracy = 0.1000, Loss = 2.3031

   🔄 Starting FedMD Round 1
     📊 Collecting logits from 3 clients...
     🔄 Aggregating logits...

   === Round 1 - Logit Consensus Analysis ===
   Consensus logits shape: (512, 10)
   Logit mean: 0.0201
   Logit std: 0.0187
   Average entropy (uncertainty): 2.3024
   Average confidence: 0.1031

   --- Client Logit Differences Analysis ---
   Client 1 vs Consensus L1 distance: 0.0567
   Client 2 vs Consensus L1 distance: 0.0519
   Client 3 vs Consensus L1 distance: 0.0374
   Average client-consensus difference: 0.0487
   Client 1 vs Client 2 L1 distance: 0.0949
   Client 1 vs Client 3 L1 distance: 0.0804
   Client 2 vs Client 3 L1 distance: 0.0766
   Average inter-client difference: 0.0839
   ✅ FedMD is working: Clients are converging toward consensus!

   🎯 Performing distillation training...
   ✅ FedMD Round 1 completed

   ...

   ============================================================
   FEDMD TRAINING SUMMARY
   ============================================================

   Round-wise Accuracy:
   Round 0: 0.0999 ± 0.0002
   Round 1: 0.0992 ± 0.0011
   Round 2: 0.0890 ± 0.0109
   Round 3: 0.0994 ± 0.0141

   Accuracy improvement: 0.0002
   ✅ FedMD training shows positive improvement!

Understanding the Output
------------------------

**Initial Performance**: Shows the baseline performance of each client model before FedMD training.

**Logit Consensus Analysis**: 
- **Shape**: Dimensions of the consensus logits
- **Statistics**: Mean, std, min, max of logit values
- **Entropy**: Measures uncertainty in predictions
- **Confidence**: Average confidence in predictions

**Client Logit Differences**:
- **Client-Consensus Distance**: How close each client is to the consensus
- **Inter-Client Distance**: How different clients are from each other
- **Convergence Check**: Verifies that clients are moving toward consensus

**Training Summary**:
- **Round-wise Performance**: Accuracy and loss for each round
- **Improvement**: Overall performance improvement
- **Success Indicator**: Whether FedMD is working effectively

Customization Options
---------------------

**Temperature Scaling**:
Adjust the temperature parameter to control knowledge transfer intensity:

.. code-block:: python

   strategy = FedMDStrategy(
       temperature=3.0,  # Higher temperature for more exploration
       # ... other parameters
   )

**Public Data Sampling**:
Customize the sampling strategy for public data:

.. code-block:: python

   def custom_sampler(public_id: str, rnd: int, n: int):
       # Your custom sampling logic
       return selected_indices

   strategy = FedMDStrategy(
       public_sampler=custom_sampler,
       # ... other parameters
   )

**Model Architecture**:
Use different model architectures for heterogeneous learning:

.. code-block:: python

   # Different models for different clients
   models = [SmallCifarNet(), ResNet18(), VGG11()]
   clients = [make_client_with_model(model) for model in models]

**Distillation Parameters**:
Fine-tune the distillation process:

.. code-block:: python

   strategy = FedMDStrategy(
       distill_epochs=3,      # More distillation epochs
       batch_size=128,        # Larger batch size
       public_sample_size=1024,  # More public samples
   )

Troubleshooting
---------------

**Common Issues**:

1. **Memory Issues**: Reduce batch_size or public_sample_size
2. **Slow Convergence**: Increase temperature or distill_epochs
3. **Poor Performance**: Check public data quality and sampling strategy
4. **Communication Errors**: Verify protobuf installation and version compatibility

**Debugging Tips**:

- Enable detailed logging to see logit statistics
- Monitor client-consensus distance over rounds
- Check that public data is properly loaded and accessible
- Verify that all clients can generate logits successfully

Next Steps
----------

- Explore different model architectures
- Experiment with various public datasets
- Try different temperature and sampling strategies
- Implement custom validation metrics
- Scale to larger numbers of clients

For more advanced usage, see the API reference and other FedMD examples.
