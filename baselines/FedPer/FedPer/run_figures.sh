#!/bin/bash

# CIFAR10 ResNet
# python -m rest2.main --config-path conf/cifar/resnet --config-name fedper_10 &&
python -m rest2.main --config-path conf/cifar/resnet --config-name fedper_8 &&
python -m rest2.main --config-path conf/cifar/resnet --config-name fedper_4 &&
python -m rest2.main --config-path conf/cifar/resnet --config-name fedavg_10 &&
python -m rest2.main --config-path conf/cifar/resnet --config-name fedavg_8 &&
python -m rest2.main --config-path conf/cifar/resnet --config-name fedavg_4 
