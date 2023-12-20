#!/bin/bash

# CIFAR10 Mobile and Resnet (non-iid n classes (FIGURE 2a&b))
for model in mobile resnet
do 
    for num_classes in 4 8 10
    do
        for algorithm in fedper fedavg
        do
           python -m fedper.main --config-path conf --config-name cifar10 dataset.num_classes=${num_classes} model_name=${model} algorithm=${algorithm}
        done
    done
done


# CIFAR10 Mobile (n head layers (FIGURE 4a))
for num_head_layers in 2 3 4
do
    python -m fedper.main --config-path conf --config-name cifar10 dataset.num_classes=4 model.num_head_layers=${num_head_layers} num_rounds=25 model_name=mobile algorithm=fedper
done
python -m fedper.main --config-path conf --config-name cifar10 num_rounds=25 model_name=mobile dataset.num_classes=4

# CIFAR10 Resnet (n head layers (FIGURE 4b))
for num_head_layers in 1 2 3
do
    python -m fedper.main --config-path conf --config-name cifar10 dataset.num_classes=4 model.num_head_layers=${num_head_layers} num_rounds=25 model_name=resnet algorithm=fedper
done
python -m fedper.main --config-path conf --config-name cifar10 num_rounds=25 model_name=resnet dataset.num_classes=4 

# FLICKR
for model in mobile resnet
do 
    python -m fedper.main --config-path conf --config-name flickr model.num_head_layers=2 model_name=${model} algorithm=fedper num_rounds=35
    python -m fedper.main --config-path conf --config-name flickr model_name=${model} algorithm=fedavg num_rounds=35
done

