#!/bin/bash

# CIFAR10 Mobile
#python -m FedPer.main --config-path conf/cifar10/mobile --config-name fedper_10 &&
#python -m FedPer.main --config-path conf/cifar10/mobile --config-name fedper_8 &&
#python -m FedPer.main --config-path conf/cifar10/mobile --config-name fedper_4 &&
#python -m FedPer.main --config-path conf/cifar10/mobile --config-name fedavg_10 &&
#python -m FedPer.main --config-path conf/cifar10/mobile --config-name fedavg_8 &&
#python -m FedPer.main --config-path conf/cifar10/mobile --config-name fedavg_4 &&

# CIFAR10 Resnet
#python -m FedPer.main --config-path conf/cifar10/resnet --config-name fedper_10 &&
#python -m FedPer.main --config-path conf/cifar10/resnet --config-name fedper_8 &&
# python -m FedPer.main --config-path conf/cifar10/resnet --config-name fedper_4 &&
#python -m FedPer.main --config-path conf/cifar10/resnet --config-name fedavg_10 &&
#python -m FedPer.main --config-path conf/cifar10/resnet --config-name fedavg_8 &&
# python -m FedPer.main --config-path conf/cifar10/resnet --config-name fedavg_4

# CIFAR10 Mobile (x head layers)
#python -m FedPer.main --config-path conf/cifar10/mobile --config-name n_2_head_layers &&
#python -m FedPer.main --config-path conf/cifar10/mobile --config-name n_3_head_layers &&
#python -m FedPer.main --config-path conf/cifar10/mobile --config-name n_4_head_layers &&
python -m FedPer.main --config-path conf/cifar10/mobile --config-name fedavg &&

# CIFAR10 Resnet (x head layers)
#python -m FedPer.main --config-path conf/cifar10/resnet --config-name n_1_head_layers &&
#python -m FedPer.main --config-path conf/cifar10/resnet --config-name n_2_head_layers &&
#python -m FedPer.main --config-path conf/cifar10/resnet --config-name n_3_head_layers &&
python -m FedPer.main --config-path conf/cifar10/resnet --config-name fedavg

# FLICKR Mobile
# python -m FedPer.main --config-path conf/flickr/mobile --config-name 2_head_layers &&
python -m FedPer.main --config-path conf/flickr/mobile --config-name fedavg &&

# FLICKR Resnet
python -m FedPer.main --config-path conf/flickr/resnet --config-name 2_head_layers &&
python -m FedPer.main --config-path conf/flickr/resnet --config-name fedavg
