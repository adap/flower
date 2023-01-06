# Starts a ray worker node and limits the number of CPU threads and GPUs available in this node.
# Use the head's ip address and port provided during the head's initialization.
# If you are running an experiment in single-node, you DON'T need to run this script.
GPUS=$(nvidia-smi --query-gpu=uuid, --format=csv,noheader | awk '{print "\"" $1 "\"" ": 1.0," }')
GPUS="{"${GPUS%?}"}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

ray start --address=${RAY_HEAD_IP}:${RAY_HEAD_PORT} --num-cpus 6

# Launch System Monitor for this node.
python launch_monitor.py 