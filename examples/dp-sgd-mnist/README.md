# Flower Example Using Tensorflow/Keras and Tensorflow Privacy

This example of Flower trains a federeated learning system where clients are free to choose 
between non-private and private optimizers. Specifically, clients can choose to train Keras models using the standard SGD optimizer or __Differentially Private__ SGD (DPSGD) from [Tensorflow Privacy](https://github.com/tensorflow/privacy). For this task we use the MNIST dataset which is split artificially among clients. This causes the dataset to be i.i.d. The clients using DPSGD track the amount of privacy spent and display it at the end of the training. 

This example is adapted from https://github.com/tensorflow/privacy/blob/master/tutorials/mnist_dpsgd_tutorial_keras.py


## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/dp-sgd-mnist . && rm -rf flower && cd dp-sgd-mnist
```

This will create a new directory called `dp-sgd-mnist` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- common.py
-- README.md
```

Project dependencies (such as `tensorflow` and `tensorflow-privacy`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run Federated Learning with TensorFlow/Keras/Tensorflow-Privacy and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
# terminal 1
poetry run python3 server.py
```
Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminals and run the following command in each:

```shell
# terminal 2
poetry run python3 client.py --partition 0
```
```shell
# terminal 3
# We will set the second client to use `dpsgd`
poetry run python3 client.py --partition 1 --dpsgd True
```

Alternatively you can run all of it in one shell as follows:

```shell
poetry run python3 server.py &
poetry run python3 client.py --partition 0 &
poetry run python3 client.py --partition 1 --dpsgd True
```

It should be noted that when starting more than 2 clients, the total number of clients you intend to run and the data partition the client is expected to use must be specified. This is because the `num_clients` is used to split the dataset.

For example, in case of 3 clients

```shell
poetry run python3 server.py --num-clients 3 &
poetry run python3 client.py --num-clients 3 --partition 0 --dpsgd True &
poetry run python3 client.py --num-clients 3 --partition 1 &
poetry run python3 client.py --num-clients 3 --partition 2 --dpsgd True
```


Additional training parameters for the client and server can be referenced by passing `--help` to either script.

Other things to note is that when all clients are running `dpsgd`, either train for more rounds or increase the local epochs to achieve optimal performance. You shall need to carefully tune the hyperparameters to your specific setup.

```shell
poetry run python3 server.py --num-clients 3  --num-rounds 20
```

```shell
poetry run python3 client.py --num-clients 3 --partition 1 --local-epochs 4 --dpsgd True
```

Running this example with 10 clients and DPSGD enabled keeping all other parameters at their default,
should converge to ~89% test set accuracy.