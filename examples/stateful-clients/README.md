# Example of Stateful Clients with Flower

This example borrows most of the code from `flower/examples/quickstart-pytorch` and showcases how to work with `stateful` clients. In this example, the client state kept track of is the time taken to run the local training state (i.e. the `fit()` method).


Running this example in itself is quite easy. Please refer to `quickstart-pytorch` for a more detailed description on how to build the environment. But in short:

1. Clone this project example:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/stateful-clients . && rm -rf flower && cd stateful-clients
```

2. Setup your Python environment with either Poetry or your fav Python env manager. 

    * For Poetry, install the environment (run `poetry install`) and activate it (run `poetry shell`), then you are ready to start the server and the clients. Note that you'll need several terminals to run this example, so make sure you do `poetry shell` in all of them to access your Python environment. 

    * If you don't want to use Poetry, use `pip install -r requirements.txt` while sourced in your environment (e.g. `conda`, `pyenv`, `virtualenv`)


## Stateful Flower Clients (gRPC)

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal. You can choose between a client with `InMemoryState` (whose state will persist for the duration of the experiment) or with an `InFileSystemState` (whose state will be loaded and stored to disk). For the latter, a subdirectory will be created in the form `./client_states/{client_id}`. There, a [python pickle](https://www.digitalocean.com/community/tutorials/python-pickle-example) file will be used to store the state. If for instance you run the experiment for a second time (assuming you don't change the path and client_id), clients will load their previously stored state automatically.  

```shell
python inmemory-client.py # or infilesystem-client.py --client_id=AN_IDENTIFIER_OF_YOUR_CHOICE
```

Start client 2 in the second terminal:

```shell
python inmemory-client.py # or infilesystem-client.py --client_id=AN_IDENTIFIER_OF_YOUR_CHOICE (but different from that of the other client)
```

You will see that PyTorch is starting a federated training. Have a look to the [Flower Quickstarter documentation](https://flower.dev/docs/quickstart-pytorch.html) for a detailed explanation.


## Stateful Flower Clients (simulation)

> This has been tested with a early version of the new [`VirtualClientEngine`](https://github.com/adap/flower/pull/1969). Therefore, `pyproject.toml` and `requirements.txt` reflect this and use `ray==2.5.1`.

For simulation with Flower's `VirtualClientEngine` you might want to still use stateful clients without having to manually handle their state. The script `simulation.py` (and the content in the `sim_utils` directory) closely resembles the [simulation-pytorch]() example. It has been extended to use stateful clients. Similarly to the example above with `gRPC` clients, the state the clients record is the time take to do `fit()`. They also use print statements to show that the state is correctly fetched and updated. In simulation, clients should only use `InMemoryClientState`. You can run this example by simply doing:

```bash
python simulation.py
```
