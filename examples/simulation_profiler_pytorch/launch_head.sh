# Starts a head node (which is also a worker node) and defines the maximum number
# of CPU threads and GPUs available in this node. Make note of the head's ip 
# address and port provided when launching this script. These will be needed if
# launching worker nodes. If you are running an experiment in single-node, 
# you just need to run this script and the launch_experiment.sh .
ray stop && ray start --head --num-cpus 6 --num-gpus 1 

# Launch System Monitor for this node.
python launch_monitor.py 