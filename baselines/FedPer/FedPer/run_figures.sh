#!/bin/bash

# CIFAR10 Mobile and Resnet (non-iid n classes (FIGURE 2a&b))
for model in mobile resnet
do 
    for num_classes in 4 8 10
    do
        python -m FedPer.main --config-path conf/cifar10/${model} --config-name fedper dataset.num_classes=${num_classes} &&
        python -m FedPer.main --config-path conf/cifar10/${model} --config-name fedavg dataset.num_classes=${num_classes} 
    done
done

# CIFAR10 Mobile (n head layers (FIGURE 4a))
#for num_head_layers in 2 3 4
#do
#    python -m FedPer.main --config-path conf/cifar10/mobile --config-name fedper model.num_head_layers=${num_head_layers} &&
#    python -m FedPer.main --config-path conf/cifar10/mobile --config-name fedavg 
#done

# CIFAR10 Resnet (n head layers (FIGURE 4b))
#for num_head_layers in 1 2 3
#do
#    python -m FedPer.main --config-path conf/cifar10/resnet --config-name fedper model.num_head_layers=${num_head_layers} &&
#    python -m FedPer.main --config-path conf/cifar10/resnet --config-name fedavg
#done

# FLICKR
#for model in mobile resnet
#do 
#    python -m FedPer.main --config-path conf/flickr/${model} --config-name fedper model.num_head_layers=2&&
#    python -m FedPer.main --config-path conf/flickr/${model} --config-name fedavg
#done

