#!/bin/sh
# Starts a head node (which is also a worker node) and defines the maximum number
# of CPU threads. The script will take care of including all available
# GPUs INDIVIDUALLY provided you have nvidia-smi installed.
# Make note of the head's ip address and port provided when launching this script. 
# These will be needed if you are launching worker nodes. However, if you are 
# running an experiment in single-node, go ahead and runlaunch_experiment.sh .
GPUS=$(nvidia-smi --query-gpu=uuid, --format=csv,noheader | awk '{print "\"" $1 "\"" ": 1.0," }')
GPUS="{"${GPUS%?}"}"

NUM_CPUS=8
STARTING_PORT=5001

WORKER_PORTS=${STARTING_PORT}
for VAR_PORT in $(seq 1 $NUM_CPUS)
do
	TEMP=`expr ${STARTING_PORT} + ${VAR_PORT} `
	WORKER_PORTS=${WORKER_PORTS}","${TEMP}
done
echo ${WORKER_PORTS}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_memory_monitor_refresh_ms=0

ray stop && ray start --head --num-cpus ${NUM_CPUS} --resources "${GPUS}" --system-config='{"object_spilling_config": "{\"type\": \"filesystem\", \"params\": {\"directory_path\": \"/hdd1/ray\", \"buffer_size\":10000000}}"}' --worker-port-list=${WORKER_PORTS}

sleep 3;

# Launch System Monitor for this node.
python launch_monitor.py 
