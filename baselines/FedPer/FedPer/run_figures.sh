#!/bin/bash

# CIFAR10 MobileNet-v1
python -m FedPer.main --config-path conf/cifar/mobile --config-name fedper_10 &&
python -m FedPer.main --config-path conf/cifar/mobile --config-name fedper_8 
