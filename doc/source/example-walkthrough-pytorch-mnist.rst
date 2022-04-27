Example Walk-Through: PyTorch & MNIST
=====================================

In this tutorial we will learn, how to train a Convolutional Neural Network on MNIST using Flower and PyTorch. 

Our example consists of one *server* and two *clients* all having the same model. 

*Clients* are responsible for generating individual weight-updates for the model based on their local datasets. 
These updates are then sent to the *server* which will aggregate them to produce a better model. Finally, the *server* sends this improved version of the model back to each *client*.
A complete cycle of weight updates is called a *round*.

Now that we have a rough idea of what is going on, let's get started. We first need to install Flower. You can do this by running :

.. code-block:: shell

  $ pip install flwr

Since we want to use PyTorch to solve a computer vision task, let's go ahead an install PyTorch and the **torchvision** library: 

.. code-block:: shell

  $ pip install torch torchvision


Ready... Set... Train!
----------------------

Now that we have all our dependencies installed, let's run a simple distributed training with two clients and one server. Our training procedure and network architecture are based on PyTorch's `Basic MNIST Example <https://github.com/pytorch/examples/tree/master/mnist>`_. This will allow you see how easy it is to wrap your code with Flower and begin training in a federated way.
We provide you with two helper scripts namely *run-server.sh* and *run-clients.sh*. Don't be afraid to look inside, they are simple enough =).

Go ahead and launch on a terminal the *run-server.sh* script first as follows:

.. code-block:: shell

  $ bash ./run-server.sh 


Now that the server is up and running, go ahead and launch the clients.  

.. code-block:: shell

  $ bash ./run-clients.sh 


Et voilà! You should be seeing the training procedure and, after a few iterations, the test accuracy for each client.

.. code-block:: shell

    Train Epoch: 10 [30000/30016 (100%)] Loss: 0.007014				
    
    Train Epoch: 10 [30000/30016 (100%)] Loss: 0.000403				
    
    Train Epoch: 11 [30000/30016 (100%)] Loss: 0.001280				
    
    Train Epoch: 11 [30000/30016 (100%)] Loss: 0.000641				
    
    Train Epoch: 12 [30000/30016 (100%)] Loss: 0.006784				
    
    Train Epoch: 12 [30000/30016 (100%)] Loss: 0.007134				
    
    Client 1 - Evaluate on 5000 samples: Average loss: 0.0290, Accuracy: 99.16%	
    
    Client 0 - Evaluate on 5000 samples: Average loss: 0.0328, Accuracy: 99.14%


Now, let's see what is really happening inside. 

Flower Server
-------------

Inside the server helper script *run-server.sh* you will find the following code that basically runs the :code:`server.py`

.. code-block:: bash 

    python -m flwr_example.quickstart_pytorch.server


We can go a bit deeper and see that :code:`server.py` simply launches a server that will coordinate three rounds of training.
Flower Servers are very customizable, but for simple workloads we can start a server using the :ref:`start_server <flwr-server-start_server-apiref>` function and leave all the configuration possibilities at their default values as seen below.

.. code-block:: python

    import flwr as fl

    fl.server.start_server(config={"num_rounds": 3})


Flower Client
-------------

Next, let's take a look at the *run-clients.sh* file. You will see that it contains a main loop that starts a set of *clients*.

.. code-block:: bash 

    python -m flwr_example.quickstart_pytorch.client \
      --cid=$i \
      --server_address=$SERVER_ADDRESS \
      --nb_clients=$NUM_CLIENTS 

* **cid**: is the client ID. It is an integer that uniquely identifies client identifier.
* **sever_address**: String that identifies IP and port of the server. 
* **nb_clients**: This defines the number of clients being created. This piece of information is not required by the client, but it helps us partition the original MNIST dataset to make sure that every client is working on unique subsets of both *training* and *test* sets.

Again, we can go deeper and look inside :code:`flwr_example/quickstart_pytorch/client.py`. 
After going through the argument parsing code at the beginning of our :code:`main` function, you will find a call to :code:`mnist.load_data`. This function is responsible for partitioning the original MNIST datasets (*training* and *test*) and returning a :code:`torch.utils.data.DataLoader` s for each of them.
We then instantiate a :code:`PytorchMNISTClient` object with our client ID, our DataLoaders, the number of epochs in each round, and which device we want to use for training (cpu or gpu).


.. code-block:: python

    client = mnist.PytorchMNISTClient(
        cid=args.cid,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        device=device,
        )

The :code:`PytorchMNISTClient` object if finally passed to :code:`fl.client.start_client` along with the server's address as the training process begins.


A Closer Look
-------------

Now, let's look closely into the :code:`PytorchMNISTClient` inside :code:`flwr_example.quickstart_pytorch.mnist` and see what it is doing:

.. code-block:: python

    class PytorchMNISTClient(fl.client.Client):
        """Flower client implementing MNIST handwritten classification using PyTorch."""
        def __init__(
            self,
            cid: int,
            train_loader: datasets,
            test_loader: datasets,
            epochs: int,
            device: torch.device = torch.device("cpu"),
        ) -> None:
            self.model = MNISTNet().to(device)
            self.cid = cid
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.device = device
            self.epochs = epochs

        def get_weights(self) -> fl.common.Weights:
            """Get model weights as a list of NumPy ndarrays."""
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def set_weights(self, weights: fl.common.Weights) -> None:
            """Set model weights from a list of NumPy ndarrays.

            Parameters
            ----------
            weights: fl.common.Weights 
                Weights received by the server and set to local model


            Returns
            -------

            """
            state_dict = OrderedDict(
                {
                    k: torch.tensor(v)
                    for k, v in zip(self.model.state_dict().keys(), weights)
                }
            )
            self.model.load_state_dict(state_dict, strict=True)

        def get_parameters(self) -> fl.common.ParametersRes:
            """Encapsulates the weight into Flower Parameters """
            weights: fl.common.Weights = self.get_weights()
            parameters = fl.common.weights_to_parameters(weights)
            return fl.common.ParametersRes(parameters=parameters)

        def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
            """Trains the model on local dataset

            Parameters
            ----------
            ins: fl.common.FitIns 
            Parameters sent by the server to be used during training. 

            Returns
            -------
                Set of variables containing the new set of weights and information the client.

            """
            weights: fl.common.Weights = fl.common.parameters_to_weights(ins.parameters)
            fit_begin = timeit.default_timer()

            # Set model parameters/weights
            self.set_weights(weights)

            # Train model
            num_examples_train: int = train(
                self.model, self.train_loader, epochs=self.epochs, device=self.device
            )

            # Return the refined weights and the number of examples used for training
            weights_prime: fl.common.Weights = self.get_weights()
            params_prime = fl.common.weights_to_parameters(weights_prime)
            fit_duration = timeit.default_timer() - fit_begin
            return fl.common.FitRes(
                parameters=params_prime,
                num_examples=num_examples_train,
                num_examples_ceil=num_examples_train,
                fit_duration=fit_duration,
            )

        def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
            """

            Parameters
            ----------
            ins: fl.common.EvaluateIns 
            Parameters sent by the server to be used during testing. 
                

            Returns
            -------
                Information the clients testing results.


The first thing to notice is that :code:`PytorchMNISTClient` instantiates a CNN model inside its constructor

.. code-block:: python

    class PytorchMNISTClient(fl.client.Client):
    """Flower client implementing MNIST handwritten classification using PyTorch."""

    def __init__(
        self,
        cid: int,
        train_loader: datasets,
        test_loader: datasets,
        epochs: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = MNISTNet().to(device)
    ...

The code for the CNN is available under :code:`quickstart_pytorch.mnist` and it is reproduced below. It is the same network found in `Basic MNIST Example <https://github.com/pytorch/examples/tree/master/mnist>`_.

.. code-block:: python

    class MNISTNet(nn.Module):
        """Simple CNN adapted from Pytorch's 'Basic MNIST Example'."""

        def __init__(self) -> None:
            super(MNISTNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x: Tensor) -> Tensor:
            """Compute forward pass.

            Parameters
            ----------
            x: Tensor 
                Mini-batch of shape (N,28,28) containing images from MNIST dataset.
                

            Returns
            -------
            output: Tensor
                The probability density of the output being from a specific class given the input.

            """
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


The second thing to notice is that :code:`PytorchMNISTClient` class inherits from the :code:`fl.client.Client` and hence it must implement the following methods:  

.. code-block:: python

    from abc import ABC, abstractmethod

    from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes


    class Client(ABC):
        """Abstract base class for Flower clients."""

        @abstractmethod
        def get_parameters(self) -> ParametersRes:
            """Return the current local model parameters."""

        @abstractmethod
        def fit(self, ins: FitIns) -> FitRes:
            """Refine the provided weights using the locally held dataset."""

        @abstractmethod
        def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            """Evaluate the provided weights using the locally held dataset."""


When comparing the abstract class to its derived class :code:`PytorchMNISTClient` you will notice that :code:`fit` calls a :code:`train` function and that :code:`evaluate` calls a :code:`test`: function. 

These functions can both be found inside the same :code:`quickstart_pytorch.mnist` module:

.. code-block:: python

    def train(
        model: torch.nn.ModuleList,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        device: torch.device = torch.device("cpu"),
    ) -> int:
        """Train routine based on 'Basic MNIST Example'

        Parameters
        ----------
        model: torch.nn.ModuleList
            Neural network model used in this example.
            
        train_loader: torch.utils.data.DataLoader
            DataLoader used in traning.
            
        epochs: int 
            Number of epochs to run in each round. 
            
        device: torch.device 
            (Default value = torch.device("cpu"))
            Device where the network will be trained within a client.

        Returns
        -------
        num_examples_train: int
            Number of total samples used during traning.

        """
        model.train()
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        print(f"Training {epochs} epoch(s) w/ {len(train_loader)} mini-batches each")
        for epoch in range(epochs):  # loop over the dataset multiple time
            print()
            loss_epoch: float = 0.0
            num_examples_train: int = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                # Grab mini-batch and transfer to device
                data, target = data.to(device), target.to(device)
                num_examples_train += len(data)

                # Zero gradients
                optimizer.zero_grad()

                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                if batch_idx % 10 == 8:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}\t\t\t\t".format(
                            epoch,
                            num_examples_train,
                            len(train_loader) * train_loader.batch_size,
                            100.0
                            * num_examples_train
                            / len(train_loader)
                            / train_loader.batch_size,
                            loss.item(),
                        ),
                        end="\r",
                        flush=True,
                    )
            scheduler.step()
        return num_examples_train


    def test(
        model: torch.nn.ModuleList,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[int, float, float]:
        """Test routine 'Basic MNIST Example'

        Parameters
        ----------
        model: torch.nn.ModuleList :
            Neural network model used in this example.
            
        test_loader: torch.utils.data.DataLoader :
            DataLoader used in test.
            
        device: torch.device :
            (Default value = torch.device("cpu"))
            Device where the network will be tested within a client.

        Returns
        -------
            Tuple containing the total number of test samples, the test_loss, and the accuracy evaluated on the test set.

        """
        model.eval()
        test_loss: float = 0
        correct: int = 0
        num_test_samples: int = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                num_test_samples += len(data)
                output = model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= num_test_samples

        return (num_test_samples, test_loss, correct / num_test_samples)


Observe that these functions basically encapsulate regular training and test loops and provide :code:`fit` and :code:`evaluate` with final statistics for each round.
You could substitute them with your own train and test loops, and also change the network architecture and the entire example would still work flawlessly. 
As a matter of fact, why not try and modify the code to an example of your liking? 



Give It a Try
-------------
Looking through the quickstart code description above will have given a good understanding on how *clients* and *servers* work in Flower, how to run a simple experiment and the internals of a client wrapper. 
Here are a few things you could try on your own and want get more experience with Flower:

- Try and change :code:`PytorchMNISTClient` so it can accept different architectures.
- Modify the :code:`train` function so that it accepts different optimizers
- Modify the :code:`test` function so that it proves not only the top-1 (regular accuracy), but also the top-5 accuracy?
- Go larger! Try to adapt the code to larger images and datasets. Why not try training on ImageNet with a ResNet-50? 

You are ready now. Enjoy learning in a federated way!
