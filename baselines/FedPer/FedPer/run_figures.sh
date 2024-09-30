#!/bin/bash

# CIFAR10 Mobile
#python -m FedPer.main --config-path conf/cifar/mobile --config-name fedper_10 &&
#python -m FedPer.main --config-path conf/cifar/mobile --config-name fedper_8 &&
#python -m FedPer.main --config-path conf/cifar/mobile --config-name fedper_4 &&
#python -m FedPer.main --config-path conf/cifar/mobile --config-name fedavg_10 &&
#python -m FedPer.main --config-path conf/cifar/mobile --config-name fedavg_8 &&
#python -m FedPer.main --config-path conf/cifar/mobile --config-name fedavg_4 &&

# CIFAR10 Resnet
#python -m FedPer.main --config-path conf/cifar/resnet --config-name fedper_10 &&
#python -m FedPer.main --config-path conf/cifar/resnet --config-name fedper_8 &&
# python -m FedPer.main --config-path conf/cifar/resnet --config-name fedper_4 &&
python -m FedPer.main --config-path conf/cifar/resnet --config-name fedavg_10 &&
python -m FedPer.main --config-path conf/cifar/resnet --config-name fedavg_8 &&
python -m FedPer.main --config-path conf/cifar/resnet --config-name fedavg_4

