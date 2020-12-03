Quickstart (PyTorch)
====================

In this tutorial we will learn how to train a Convolutional Neural Network on MNIST using Flower and PyTorch. 

First of all, it is recommended to create a virtual environment and run everything within a `virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_. 

Our example consists of one *server* and two *clients* all having the same model. 

*Clients* are responsible for generating individual weight-updates for the model based on their local datasets. 
These updates are then sent to the *server* which will aggregate them to produce a better model. Finally, the *server* sends this improved version of the model back to each *client*.
A complete cycle of weight updates is called a *round*.

Now that we have a rough idea of what is going on, let's get started. We first need to install Flower. You can do this by running :

.. code-block:: shell

  $ pip install flwr

Since we want to use PyTorch to solve a computer vision task, let's go ahead and install PyTorch and the **torchvision** library: 

.. code-block:: shell

  $ pip install torch torchvision


Run it with a shell script
--------------------------

Now that we have all our dependencies installed, let's run a simple distributed training with two clients and one server. Our training procedure and network architecture are based on PyTorch's `Basic MNIST Example <https://github.com/pytorch/examples/tree/master/mnist>`_. This will allow you see how easy it is to wrap your code with Flower and begin training in a federated way.
You can use two helper scripts namely :code:`run-server.sh` and :code:`run-clients.sh`. 
First, create the :code:`run-server.sh`:

.. code-block:: shell

    python -m server

and make the script executable: 

.. code-block:: shell

    $ bash chmod +x ./run-server.sh


Second, create :code:`run-client.sh`:

.. code-block:: shell

    set -e
    SERVER_ADDRESS="[::]:8080"
    NUM_CLIENTS=2
    echo "Starting $NUM_CLIENTS clients."
    for ((i = 0; i < $NUM_CLIENTS; i++))
    do
        echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
        python -m client \
            --cid=$i \
            --server_address=$SERVER_ADDRESS \
            --nb_clients=$NUM_CLIENTS &
    done
    echo "Started $NUM_CLIENTS clients."

and make it as well executable:

.. code-block:: shell

    $ bash chmod +x ./run-client.sh

The script contains a main loop to start a set of :code:`NUM_CLIENTS` clients. Here  you can set how many clients participating on the federated learning workload. The clients are labeled by a counter :code:`--cid` for identification. In order to connect each client to the server the :code:`SERVER_ADDRESS` can be set or a default value of :code:`[::]:8080` can be used. 

Create a server
---------------

Before you can run both scripts you need to create :code:`server.py` and :code:`client.py`. 
Let's start with :code:`server.py` since it only requires the flwr package and starts the flower server by using only one command. 

.. code-block:: python

    import flwr as fl

    fl.server.start_server(config={"num_rounds": 3})

Create some clients
-------------------

The client script is longer but consists mostly of settings that you may want to adjust later to change your federated learning setup. 
The :code:`client.py` needs a few packages as numpy, pytorch, flower  and of course the data sample of MNIST. 

.. code-block:: python

    from argparse import ArgumentParser

    import numpy as np
    import torch

    import flwr as fl

    from flwr_example.quickstart_pytorch import mnist

    DATA_ROOT = "./data/mnist"

    if __name__ == "__main__":
        # Training settings
        parser = ArgumentParser(description="PyTorch MNIST Example")
        parser.add_argument(
            "--server_address",
            type=str,
            default="[::]:8080",
            help=f"gRPC server address (default: '[::]:8080')",
        )
        parser.add_argument(
            "--cid",
            type=int,
            metavar="N",
            help="ID of current client (default: 0)",
        )
        parser.add_argument(
            "--nb_clients",
            type=int,
            default=2,
            metavar="N",
            help="Total number of clients being launched (default: 2)",
        )
        parser.add_argument(
            "--train-batch-size",
            type=int,
            default=64,
            metavar="N",
            help="input batch size for training (default: 64)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=1000,
            metavar="N",
            help="input batch size for testing (default: 1000)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=14,
            metavar="N",
            help="number of epochs to train (default: 14)",
        )

        args = parser.parse_args()

        # Load MNIST data
        train_loader, test_loader = mnist.load_data(
            data_root=DATA_ROOT,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            cid=args.cid,
            nb_clients=args.nb_clients,
        )

        # pylint: disable=no-member
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # pylint: enable=no-member

        # Instantiate client
        client = mnist.PytorchMNISTClient(
            cid=args.cid,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            device=device,
        )

        # Start client
        fl.client.start_client(args.server_address, client)

With only 4 scripts you are ready to run your first federated MNIST workload. You just need to start the server:

.. code-block:: shell

  $ bash ./run-server.sh 

and in a second terminal you need to start the clients:

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

Closer look at the server
-------------------------

The :code:`server.py` simply launches a server that will coordinate three rounds of training.
Flower Servers are very customizable, but for simple workloads we can start a server and leave all the configuration possibilities at their default values.

Closer look at the client
-------------------------

Next, let's take a look at the client part that is more complex since the training of the MNIST data happens here.
Again, we can go deeper and look inside :code:`client.py`. You find many parameters to setup your own federated learning workload:

#. :code:`--server_address` 
    * setup your server address to connect the clients to server.
#. :code:`--cid`     
    * counter to identify all clients
#. :code:`--nb_clients`  
    * set the number of clients connected to one server
#. :code:`--train-batch-size`    
    * set up the size of the training batch for each client
#. :code:`--test-batch-size`     
    * set up the size of the test batch
#. :code:`--epochs`  
    * set up the number of epochs to run for each client

Play a bit around with the settings to get a feeling of a federated learning setup. 

After going through the argument parsing code at the beginning of our function, you will find a call to :code:`mnist.load_data`.

.. code-block:: python

    # Load MNIST data
    train_loader, test_loader = mnist.load_data(
        data_root=DATA_ROOT,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        cid=args.cid,
        nb_clients=args.nb_clients,
    )

This function is responsible for partitioning the original MNIST datasets (*training* and *test*) and returning a :code:`torch.utils.data.DataLoader` s for each of them.
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

Now, let's look closely into the :code:`PytorchMNISTClient`. As soon as you install the *flwr* package you also install *flwr_example* where you can find :code:`flwr_example.quickstart_pytorch.mnist`. If you run already the Keras example then the code will be familiar to you:

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

            state_dict = OrderedDict(
                {
                    k: torch.Tensor(v)
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
            """Trains the model on local dataset"""

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
            weights = fl.common.parameters_to_weights(ins.parameters)

            # Use provided weights to update the local model
            self.set_weights(weights)

            (
                num_examples_test,
                test_loss,
                accuracy,
            ) = test(self.model, self.test_loader, device=self.device)
            print(
                f"Client {self.cid} - Evaluate on {num_examples_test} samples: Average loss: {test_loss:.4f}, Accuracy: {100*accuracy:.2f}%\n"
            )

            # Return the number of evaluation examples and the evaluation result (loss)
            return fl.common.EvaluateRes(
                num_examples=num_examples_test,
                loss=float(test_loss),
                accuracy=float(accuracy),
            )

The code contains 5 main functions similar to the Keras example. 

#. :code:`get_weights`
    * receive the model weights calculated by the local model
#. :code:`set_weights`
    * set the model weights on the local model that are received from the server
#. :code:`get_parameters`
    * encapsulates the weight into Flower parameters
#. :code:`fit`
    * set the local model weights
    * train the local model
    * receive the updated local model weights
#. :code:`evaluate`
    * test the local model 

The fitting function trains the MNIST dataset with a typical CNN that can be found in the `Example Walk-Through: PyTorch & MNIST <https://flower.dev/docs/example_walkthrough_pytorch_mnist.html>`_ .
Observe that these functions basically encapsulate regular training and test loops and provide :code:`fit` and :code:`evaluate` with final statistics for each round.
You could substitute them with your own train and test loops, and also change the network architecture and the entire example would still work flawlessly. 
As a matter of fact, why not try and modify the code to an example of your liking? 
