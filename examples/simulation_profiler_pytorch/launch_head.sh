# Starts a head node and defines the maximum of CPU threads and GPUs available for this node.
# Make note of the head's ip address and port. This will be needed by the worker nodes.
# If you are running an experiment in single-node, you just need to run this script.
ray stop && ray start --head --num-cpus 6 --num-gpus 1 

# Launch System Monitor for this node.
python launch_monitor.py 

# This node will also run the aggregation server, which will orchestrate the training
python main.py 

# Stop System Monitor
ray stop 