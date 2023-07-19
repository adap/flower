# Example of Stateful Clients with Flower

This example borrows most of the code from `flower/examples/quickstart-pytorch` and showcases how to work with `stateful` clients. In this example, the client state kept track of is the time taken to run the local training state (i.e. the `fit()` method.)
Running this example in itself is quite easy. Please refer to `quickstart-pytorch` for a more detailed description on how to build the environment. But in short:

1. Clone this project example:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/stateful-clients . && rm -rf flower && cd stateful-clients
```

2. Setup your Python environment with either Poetry or your fav Python env manager. 

    * For Poetry, install the environment (run `poetry install`) and activate it (run `poetry shell`), then you are ready to start the server and the clients. Note that you'll need several terminals to run this example, so make sure you do `poetry shell` in all of them to access your Python environment. 

    * If you don't want to use Poetry, use `pip install -r requirements.txt` while sourced in your environment (e.g. `conda`, `pyenv`, `virtualenv`)


## Run Federated Learning with PyTorch and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal. You can choose between a client with `InMemoryState` (whose state will persist for the duration of the experiment) or with an `InFileSystemState` (whos state will be loaded and stored to disk)

```shell
python inmemory-client.py # or infilesystem-client.py --client_id=AN_IDENTIFIER_OF_YOUR_CHOICE
```

Start client 2 in the second terminal:

```shell
python inmemory-client.py # or infilesystem-client.py --client_id=AN_IDENTIFIER_OF_YOUR_CHOICE (but different from that of the other client)
```

You will see that PyTorch is starting a federated training. Have a look to the [Flower Quickstarter documentation](https://flower.dev/docs/quickstart-pytorch.html) for a detailed explanation.
