device="cpu"
total_gpus=1
total_cpus=30
client_cpus=1
client_gpu=0.05


check_cache=True          
epochs=5,10
mnames=lenet
num_clients=50,30
multirun_faulty_clients=5
dataset=cifar10,mnist


poetry run python -m feddebug.main --multirun device=$device total_gpus=$total_gpus total_cpus=$total_cpus client_cpus=$client_cpus client_gpu=$client_gpu check_cache=$check_cache data_dist.dist_type=iid num_clients=$num_clients model.name=$mnames dataset.name=$dataset client.epochs=$epochs multirun_faulty_clients=$multirun_faulty_clients

