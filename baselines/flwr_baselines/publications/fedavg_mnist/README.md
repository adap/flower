### Dataset

The dataset used will be the one provided by torchvision.datasets.MNIST

### Partitioning

Two methods of partitioning will be used, mimicking what is described in the paper.

The IID method will just create a random split of the data amongst the clients.

The non-IID method will first sort the data using the labels and distribute the data to each client by chunks of a given size.

### Model

The model used will be the one as described by the paper : *“A CNN with two 5x5 convolution layers (the first with
32 channels, the second with 64, each followed with 2x2
max pooling), a fully connected layer with 512 units and
ReLu activation, and a final softmax output layer (1,663,370
total parameters).”*

### Client number

In the paper, the number of clients used is 100. For the moment, 10 clients will be used in order to speed up the tests.

### Hyperparameters

The hyperparameters used will be the following :

* Batch size (32)
* Validation set percentage (10% of training set)
* The fraction of available clients for training (fraction_fit) will probably be varied between 0.1 and 1.
* For evaluation, the fraction of available clients (fraction_evaluate) will be arbitrarily set to 0.5 (same as the Flower tutorial).
* The number of rounds will also be arbitrarily set to 5 (same as the Flower tutorial).

### FL Algorithm

To mimic the paper, fl.server.strategy.FedAvg will be used.

### Structure

Not sure about folders...

datasets.py -> will probably contain a function that returns DataLoaders

model.py -> defines the model architecture, the train and test functions

client.py -> defines the FlowerClient and a function to instantiate it

main.py -> where the server logic is defined and everything is ran
