# Flower Example using PyTorch (SLURM)

> This example borrows most of its content from the [`quickstart-pytorch`](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example. The main difference here is in regards with the python environment setup (using conda) and the deployment infrastructure (SLURM). This Flower FL pipeline is the same as in the aforementioned example.

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/jafermarq/flower.git && mv flower/examples/quickstart-pytorch-slurm . && rm -rf flower && cd quickstart-pytorch-slurm
```

This will create a new directory called `quickstart-pytorch-slurm` containing the following files:

```shell
-- slurm.sh
-- requirements.txt
-- client.py
-- server.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `requirements.txt`. In the [CSD3](https://docs.hpc.cam.ac.uk/hpc/index.html) there is no poetry, so conda is your best option:

```bash
# create a new environment
conda create -n flower-slurm python=3.9

# activate it
source activate flower-slurm

# install dependencies
pip install -r requirements.txt
```


## Run Federated Learning with PyTorch and Flower (in SLURM)

If you have run other examples of Flower without using the [VirtualClientEngine](https://flower.dev/docs/framework/how-to-run-simulations.html), you already know the drill, you need to manually launch the `server` and then connect to it as many `clients` as you wish. This can be done easily by ssh-ing into each device if you have access to it, or if you want to test things on your development machine you could simply open different terminal shells (maybe via [tmux](https://github.com/tmux/tmux/wiki#welcome-to-tmux)). In order to do the same under SLURM, we need a bit more work.

The best is to define a standard SLURM script that we can executed via `sbatch` command. The basic idea is as follows: We want to submit to the SLURM scheduler N nodes (1 for the server, and N-1 for clients). But how to do this from a single script? See below the content of `slurm.sh`:

```bash
#!/bin/bash
# shellcheck disable=SC2206
#SBATCH -A <YOUR-PROJECT-CODE>
#SBATCH --partition=<YOUR-FAVOURITE-PARTITION> (for this example 'cclake' is fine)
#SBATCH --ntasks-per-node=1 # important, else it will spawn the same task N times.
#SBATCH --nodes=3 # N (in this case 3 so it will use 1 task for Server, and 2 nodes one for each client)
#SBATCH --time=0:03:00 # for the workload in this example, we don't need much time to complete it. Providing a time helps SLURM schedule your workload ahead of others (potentially -- but usually true if you have short jobs).

# Source your environment
source activate flower-slurm

# Then first node is going to be the Flower server
# we need to capture its IP so we can connect the
# Flower clients to the server
ip=$(hostname --ip-address)

# Let's get a list of nodes that have been assigned to our experiment
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

worker_num=$((SLURM_JOB_NUM_NODES - 1)) # number of nodes other than the server node
# Spawn a client in each node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting Client $i at $node_i"
  # launch clients but delay call to python client (so there is time for the server to start up)
  srun --nodes=1 --ntasks=1 -w "$node_i" python client.py --server_address $ip --wait_for_server 15 &
done

# Launch server
echo "Starting server at $ip"
python server.py --server_address $ip

```

Ensure you have updated the required header fields in the `slurm.sh`. Then executed it as follows after you have created your conda environment:


```bash
sbatch slurm.sh
```

The first time you run this, the clients will download the `MNIST` dataset to insider this example's directory if it's not found. For larger datasets, you'd normally download them first to an appropiate location and then set the path to where the data is inside the Flower clients.

The SLURM log for this example will look something like the one below. 2 Flowe clients train a small CNN for 3 FL rounds on MNIST:

```bash
Starting Client 1 at <CSD3 hostname> (redacted)
Starting Client 2 at <CSD3 hostname> (redacted)
Starting server at <CSD3 Flower server ip> (redacted)

# Server start
INFO flwr 2023-09-27 15:38:31,956 | app.py:162 | Starting Flower server, config: ServerConfig(num_rounds=3, round_timeout=None)
INFO flwr 2023-09-27 15:38:31,979 | app.py:175 | Flower ECE: gRPC server running (3 rounds), SSL is disabled
INFO flwr 2023-09-27 15:38:31,979 | server.py:89 | Initializing global parameters
INFO flwr 2023-09-27 15:38:31,979 | server.py:276 | Requesting initial parameters from one random client

# now clients download MNIST, then connect to Flower server

INFO flwr 2023-09-27 15:38:51,593 | grpc.py:49 | Opened insecure gRPC connection (no certificates were passed)
DEBUG flwr 2023-09-27 15:38:51,595 | connection.py:42 | ChannelConnectivity.IDLE
DEBUG flwr 2023-09-27 15:38:51,595 | connection.py:42 | ChannelConnectivity.CONNECTING
DEBUG flwr 2023-09-27 15:38:51,596 | connection.py:42 | ChannelConnectivity.READY
INFO flwr 2023-09-27 15:38:51,605 | server.py:280 | Received initial parameters from one random client
INFO flwr 2023-09-27 15:38:51,605 | server.py:91 | Evaluating initial parameters
INFO flwr 2023-09-27 15:38:51,605 | server.py:104 | FL starting
INFO flwr 2023-09-27 15:38:51,665 | grpc.py:49 | Opened insecure gRPC connection (no certificates were passed)
DEBUG flwr 2023-09-27 15:38:51,667 | connection.py:42 | ChannelConnectivity.IDLE
DEBUG flwr 2023-09-27 15:38:51,667 | connection.py:42 | ChannelConnectivity.CONNECTING
DEBUG flwr 2023-09-27 15:38:51,671 | connection.py:42 | ChannelConnectivity.READY
DEBUG flwr 2023-09-27 15:38:51,671 | server.py:222 | fit_round 1: strategy sampled 2 clients (out of 2)
100%|██████████| 1875/1875 [00:17<00:00, 109.23it/s]3it/s]t/s] [00:12<00:04, 107.81it/s]/s]
100%|██████████| 1875/1875 [00:17<00:00, 107.05it/s]
DEBUG flwr 2023-09-27 15:39:09,199 | server.py:236 | fit_round 1 received 2 results and 0 failures
WARNING flwr 2023-09-27 15:39:09,203 | fedavg.py:242 | No fit_metrics_aggregation_fn provided
DEBUG flwr 2023-09-27 15:39:09,203 | server.py:173 | evaluate_round 1: strategy sampled 2 clients (out of 2)
100%|██████████| 10000/10000 [00:08<00:00, 1222.54it/s]it/s]10000 [00:05<00:03, 1175.97it/s]
100%|██████████| 10000/10000 [00:08<00:00, 1209.00it/s]
DEBUG flwr 2023-09-27 15:39:17,481 | server.py:187 | evaluate_round 1 received 2 results and 0 failures
DEBUG flwr 2023-09-27 15:39:17,481 | server.py:222 | fit_round 2: strategy sampled 2 clients (out of 2)
100%|██████████| 1875/1875 [00:17<00:00, 110.11it/s]8.38it/s]]0:12<00:04, 108.83it/s]1it/s]
100%|██████████| 1875/1875 [00:17<00:00, 107.74it/s]
DEBUG flwr 2023-09-27 15:39:34,890 | server.py:236 | fit_round 2 received 2 results and 0 failures
DEBUG flwr 2023-09-27 15:39:34,892 | server.py:173 | evaluate_round 2: strategy sampled 2 clients (out of 2)
100%|██████████| 10000/10000 [00:08<00:00, 1221.95it/s]0, 1219.76it/s], 1158.26it/s]
100%|██████████| 10000/10000 [00:08<00:00, 1208.11it/s]
DEBUG flwr 2023-09-27 15:39:43,176 | server.py:187 | evaluate_round 2 received 2 results and 0 failures
DEBUG flwr 2023-09-27 15:39:43,176 | server.py:222 | fit_round 3: strategy sampled 2 clients (out of 2)
100%|██████████| 1875/1875 [00:17<00:00, 108.94it/s]07.25it/s]7.98it/s]07.89it/s]
100%|██████████| 1875/1875 [00:17<00:00, 107.74it/s]
DEBUG flwr 2023-09-27 15:40:00,585 | server.py:236 | fit_round 3 received 2 results and 0 failures
DEBUG flwr 2023-09-27 15:40:00,587 | server.py:173 | evaluate_round 3: strategy sampled 2 clients (out of 2)
100%|██████████| 10000/10000 [00:08<00:00, 1233.73it/s]0000 [00:05<00:03, 1225.80it/s]
100%|██████████| 10000/10000 [00:08<00:00, 1220.38it/s]
DEBUG flwr 2023-09-27 15:40:08,787 | server.py:187 | evaluate_round 3 received 2 results and 0 failures
INFO flwr 2023-09-27 15:40:08,787 | server.py:153 | FL finished in 77.1818270306103

# FL finished, now server outputs some statistics

INFO flwr 2023-09-27 15:40:08,787 | app.py:225 | app_fit: losses_distributed [(1, 1523.18408203125), (2, 817.1888427734375), (3, 666.5919189453125)]
INFO flwr 2023-09-27 15:40:08,787 | app.py:226 | app_fit: metrics_distributed_fit {}
INFO flwr 2023-09-27 15:40:08,787 | app.py:227 | app_fit: metrics_distributed {'accuracy': [(1, 0.9557), (2, 0.9752), (3, 0.9788)]}
INFO flwr 2023-09-27 15:40:08,787 | app.py:228 | app_fit: losses_centralized []
INFO flwr 2023-09-27 15:40:08,787 | app.py:229 | app_fit: metrics_centralized {}

# Server tells clients to shutdown

DEBUG flwr 2023-09-27 15:40:08,790 | connection.py:139 | gRPC channel closed
INFO flwr 2023-09-27 15:40:08,790 | app.py:215 | Disconnect and shut down
DEBUG flwr 2023-09-27 15:40:08,790 | connection.py:139 | gRPC channel closed
INFO flwr 2023-09-27 15:40:08,790 | app.py:215 | Disconnect and shut down
```
