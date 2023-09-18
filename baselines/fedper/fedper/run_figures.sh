#!/bin/bash

# CIFAR10 Mobile and Resnet (non-iid n classes (FIGURE 2a&b))
#for model in mobile resnet
#do 
#    for num_classes in 4 8 10
#    do
#        python -m fedper.main --config-path conf/cifar10/${model} --config-name fedper dataset.num_classes=${num_classes} num_rounds=50 &&
#        python -m fedper.main --config-path conf/cifar10/${model} --config-name fedavg dataset.num_classes=${num_classes} num_rounds=50
#    done
#done  

#python -m fedper.main --config-path conf/cifar10/resnet --config-name fedper dataset.num_classes=8 num_rounds=50 &&
#python -m fedper.main --config-path conf/cifar10/resnet --config-name fedper dataset.num_classes=4 num_rounds=50 &&
#python -m fedper.main --config-path conf/cifar10/resnet --config-name fedavg dataset.num_classes=10 num_rounds=50 &&
#python -m fedper.main --config-path conf/cifar10/resnet --config-name fedavg dataset.num_classes=8 num_rounds=50 &&
#python -m fedper.main --config-path conf/cifar10/resnet --config-name fedavg dataset.num_classes=4 num_rounds=50 

# CIFAR10 Mobile (n head layers (FIGURE 4a))
#for num_head_layers in 2 3 4
#do
#    python -m fedper.main --config-path conf/cifar10/mobile --config-name fedper model.num_head_layers=${num_head_layers} num_rounds=25 &&
#    python -m fedper.main --config-path conf/cifar10/mobile --config-name fedavg num_rounds=25
#done

# CIFAR10 Resnet (n head layers (FIGURE 4b))
for num_head_layers in 1 2 3
do
    # python -m fedper.main --config-path conf/cifar10/resnet --config-name fedper model.num_head_layers=${num_head_layers} num_rounds=25
    # python -m fedper.main --config-path conf/cifar10/resnet --config-name fedavg
done

# FLICKR
#for model in mobile resnet
#do 
#    python -m fedper.main --config-path conf/flickr/${model} --config-name fedper model.num_head_layers=2&&
#    python -m fedper.main --config-path conf/flickr/${model} --config-name fedavg
#done

