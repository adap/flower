# Flower end-to-end tests

This directory contains folders for different scenarios that need to be tested and validated before a change can be added to Flower.

Currently the following end-to-end tests have been implemented:

* [bare](adap/flower/tests/bare): testing Flower in a bare minimum scenario, that is, with a dummy model and dummy operations. This is mainly to test the core functionnality of Flower independently from any framework. It can easily be extendended to test more complex communication set-ups

* [pytorch](adap/flower/tests/pytorch): testing Flower with PyTorch by using the CIFAR10 dataset and a CNN.

* [tensorflow](adap/flower/tests/tensorflow): testing Flower with Tensorflow by using the CIFAR10 dataset and a MobileNetV2 model.
