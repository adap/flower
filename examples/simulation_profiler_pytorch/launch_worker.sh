# Starts a ray worker node and limits the number of CPU threads and GPUs available in this node.
# Use the head's ip address and port provided during the head's initialization.
# If you are running an experiment in single-node, you DON'T need to run this script.
ray start --address=${RAY_HEAD_IP}:${RAY_HEAD_PORT} --num-cpus 6 --num-gpus 1

# Launch System Monitor for this node.
python launch_monitor.py 