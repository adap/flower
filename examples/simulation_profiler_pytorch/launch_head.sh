#!/bin/sh
# Starts a head node (which is also a worker node) and defines the maximum number
# of CPU threads. The script will take care of including all available
# GPUs INDIVIDUALLY provided you have nvidia-smi installed.
# Make note of the head's ip address and port provided when launching this script. 
# These will be needed if you are launching worker nodes. However, if you are 
# running an experiment in single-node, go ahead and runlaunch_experiment.sh .
GPUS=$(nvidia-smi --query-gpu=uuid, --format=csv,noheader | awk '{print "\"" $1 "\"" ": 1.0," }')
GPUS="{"${GPUS%?}"}"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_memory_monitor_refresh_ms=0

ray stop && ray start --head --num-cpus 8 --resources "${GPUS}"

# Launch System Monitor for this node.
python launch_monitor.py 
