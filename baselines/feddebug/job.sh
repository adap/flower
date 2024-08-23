check_cache=True          
epochs=5,10,15,20
mnames=densenet121,resnet50,resnet18,lenet
num_clients=30,50
multirun_faulty_clients=2,5,7
dataset=mnist,cifar10
poetry run python -m feddebug.main --multirun check_cache=$check_cache data_dist.dist_type=iid num_clients=$num_clients model.name=$mnames dataset.name=$dataset client.epochs=$epochs multirun_faulty_clients=$multirun_faulty_clients