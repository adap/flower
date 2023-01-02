# Start ray head with a maximum of 3 workers, Maximum 1 GPU being used
ray stop && ray start --head --num-cpus 8 --num-gpus 1 # LIMIT TOTAL AMOUNT OF RESOURCES

# Launch System Monitor. Notice that you should run one monitor per node.
python launch_monitor.py 

# Launch experiment
python main.py --num_cpus_per_client 1

# Stop System Monitor
ray stop 