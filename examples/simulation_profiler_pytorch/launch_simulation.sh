# This script launches the actual FL training.
# Make sure you have initialized the head node and
# eventual worker nodes before launching this script.
python main.py 

# Once finished, stop System Monitor for this node.
ray stop 