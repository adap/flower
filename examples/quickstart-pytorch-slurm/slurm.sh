#!/bin/bash
# shellcheck disable=SC2206
#SBATCH -A <YOUR-PROJECT-CODE>
#SBATCH --partition=cclake
#SBATCH --ntasks-per-node=1 # important, else it will spawn the same task N times.
#SBATCH --nodes=3
#SBATCH --time=0:03:00

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
