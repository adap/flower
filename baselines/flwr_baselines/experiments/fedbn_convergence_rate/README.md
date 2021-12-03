# FedBN Baseline - Convergence Rate

## Experiment Introduction

The **FedBN - Convergence Rate** baseline is based on the paper [FEDBN: FEDERATED LEARNING ON NON-IID FEATURES VIA LOCAL BATCH NORMALIZATION](https://arxiv.org/pdf/2102.07623.pdf) and reproduces the results being presented in *Chapter 5 - Convergence Rate (Figure 3)* by running Flower as the federated learning framework. This experiment uses 5 completly different image datasets of digits to emulate a non-IID data distribution between the clients. Therefore, 5 clients are used for the training. The local training is setup with 1 epoch and a CNN model is used together with the SGD optimizer. The loss is calculated by the cross entropy loss. 

## Dataset 

### General Overview

5 different data sets are used to simulate a non-IID data distribution within 5 clients. The following datasets are used:

* [MNIST](https://ieeexplore.ieee.org/document/726791)
* [MNIST-M]((https://arxiv.org/pdf/1505.07818.pdf))
* [SVHN](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
* [USPS](https://ieeexplore.ieee.org/document/291440)
* [SynthDigits](https://arxiv.org/pdf/1505.07818.pdf)

A more detailed explanation of the datasets are given in the following table. 

|     | MNIST     | MNIST-M   | SVHN  |  USPS    | SynthDigits |
|--- |---        |---        |---    |---            |---    |
| data type| handwritten digits| MNIST modification randomly colored with colored patches| Street view house numbers | handwritten digits from envelopes by the U.S. Postal Service | Syntehtic digits Windows TM font varying the orientation, blur and stroke colors |
| color | greyscale | RGB | RGB | greyscale | RGB |
| pixelsize | 28x28 | 28 x 28 | 32 x32 | 16 x16 | 32 x32 |
| labels | 0-9 | 0-9 | 1-10 | 0-9 | 1-10 |
| number of trainset | 60.000 | 60.000 | 73.257 | 9,298 | 50.000 |
| number of testset| 10.000 | 10.000 | 26.032 | - | 0 |
| image shape | (28,28) | (28,28,3) | (32,32,3) | (16,16) | (32,32,3) |

### Dataset Download

The Research team from the [FedBN paper](https://arxiv.org/pdf/2102.07623.pdf) prepared a pre-processed dataset on their GitHub repository that is available [here](https://github.com/med-air/FedBN). Please download their data, save it in a `/data` directory and unzip afterwards. 
The training data contains only 7438 samples and is splitted in 10 files but only one file is used for **FedBN: Convergence Rate** baseline. Therefore, 743 samples are used for the local training. 

## Training Setup

### CNN Architecture

The CNN architecture is given in the paper and reused to create the **FedBN - Convergence Rate** baseline.

| Layer | Details| 
| ----- | ------ |
| 1 | Conv2D(3,64, 5,1,2) <br> BN(64), ReLU, MaxPool2D(2,2)  |
| 2 | Conv2D(64, 64, 5, 1, 2) <br> BN(64), ReLU, MaxPool2D(2,2) |
| 3 | Conv2D(64, 128, 5, 1, 2) <br> BN(128), ReLU |
| 4 | FC(6272, 2048) <br> BN(2048), ReLU |
| 5 | FC(2048, 512) <br> BN(512), ReLU |
| 6 | FC(512, 10) |

### Training Paramater

| Description | Value |
| ----------- | ----- |
| training samples | 743 |
| mu | 10E-2 |
| local epochs | 1 |
| loss | cross entropy loss |
| optimizer | SGD |

## Running the Experiment

Before you run any program of the baseline experiment, please get the required data and place it in the `/data` directory. 

As soons as you have downloaded the data you are ready to start the baseline experiment. The baseline contains different programms:

* utils/data_utils.py
* cnn_model.py
* client.py
* server.py 
* run.sh

In order to run the experiment you simply make `run.sh` executable and run it. The `run.sh` creates first the files where the evaluation results are saved and starts the `server.py` and 5 clients in parallel with `client.py`. Each client loads another dataset as explained before. The clients saves the training and evaluation parameters in a dict with the following informations:

```python
train_dict = {"dataset": self.num_examples["dataset"], "fl_round" : fl_round, "strategy": self.mode , "train_loss": loss, "train_accuracy": accuracy}
```
```python
test_dict =  {"dataset": self.num_examples["dataset"], "fl_round" : fl_round, "strategy": self.mode, "test_loss": loss, "test_accuracy": accuracy}
```

The `utils/data_utils.py` prepares/loads the data for the training and `cnn_model.py` set up the CNN model architecture.   
