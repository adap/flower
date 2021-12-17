# FedBN Baseline - Convergence Rate

## Experiment Introduction

The **FedBN - Convergence Rate** baseline is based on the paper [FEDBN: FEDERATED LEARNING ON NON-IID FEATURES VIA LOCAL BATCH NORMALIZATION](https://arxiv.org/pdf/2102.07623.pdf) and reproduces the results being presented in *Chapter 5 - Convergence Rate (Figure 3)* by running Flower as the federated learning framework. This experiment uses 5 completely different image datasets of digits to emulate a non-IID data distribution between the clients. Therefore, 5 clients are used for the training. The local training is set up with 1 epoch and a CNN model is used together with the SGD optimizer. The loss is calculated by the cross-entropy loss. 

## Dataset 

### General Overview

5 different data sets are used to simulate a non-IID data distribution within 5 clients. The following datasets are used:

* [MNIST](https://ieeexplore.ieee.org/document/726791)
* [MNIST-M]((https://arxiv.org/pdf/1505.07818.pdf))
* [SVHN](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
* [USPS](https://ieeexplore.ieee.org/document/291440)
* [SynthDigits](https://arxiv.org/pdf/1505.07818.pdf)

A more detailed explanation of the datasets is given in the following table. 

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

The Research team from the [FedBN paper](https://arxiv.org/pdf/2102.07623.pdf) prepared a pre-processed dataset on their GitHub repository that is available [here](https://github.com/med-air/FedBN). Please download their data, save it in a `/data` directory and unzip afterward. 
The training data contains only 7438 samples and is split into 10 files but only one file is used for **FedBN: Convergence Rate** baseline. Therefore, 743 samples are used for the local training. 

For the original data, please run the folloing to download and perform preprocessing:
```bash
# download data (will create a directory in ./path)
python utils/data_download.py
# preprocess
python utils/data_preprocess.py
```

All the datasets (with the exception of SynthDigits) can be downloaded from the original sources:

```bash
# download
python utils/data_download_raw.py # then run `data_preprocess.py` as before.
```

## Training Setup ##

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

As soon as you have downloaded the data you are ready to start the baseline experiment. The baseline contains different programs:

* utils/data_utils.py
* cnn_model.py
* client.py
* server.py 
* run.sh

In order to run the experiment, you simply make `run.sh` executable and run it. 

```bash
chmod +x run.sh
# now compare fedavg vs fedbn
./run.sh fedavg
# you might want ot wait for the above to finish before running in FedBN mode.
./run.sh fedbn
```

The `run.sh` creates first the files where the evaluation results are saved and starts the `server.py` and 5 clients in parallel with `client.py`. Each client loads another dataset as explained before. The clients save the evaluation results after the parameters are sent from the server to the client and right before the local training. The saved parameters are included in a dict with the following information:

```python
test_dict =  {"dataset": self.num_examples["dataset"], "fl_round" : fl_round, "strategy": self.mode, "test_loss": loss, "test_accuracy": accuracy}
```

The `utils/data_utils.py` prepares/loads the data for the training and `cnn_model.py` set up the [CNN model architecture](#cnn-architecture). This baseline only takes one single file with 743 samples from the downloaded dataset. 

### Server

This baseline compares the Federate Averaging (FedAvg) with Federated Batch Normalization (FedBN). In both cases, we are using the FedAvg on the server-side. All parameters being created in the model architecture are sent from the client to the server and aggregated. However, in the case of FedBN, we are setting up the client to exclude the BN layer from the transmission to the server. FedBN is therefore a strategy that is on the client-side. 
The server is kept very simple and the same for both settings. We are using FedAvg on the server-side with the parameters `min_fit_clients`, `min_eval_clients`, and `min_available_clients` that are set to the value `5` since we have five clients to be trained and evaluated in each FL round. All in all, the *FedBN* paper runs 600 FL rounds that can be set up correspondingly.     

```python
import flwr as fl

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=5,
        min_eval_clients=5,
        min_available_clients=5,
    )
    fl.server.start_server("[::]:8080", config={"num_rounds": 100}, strategy=strategy)

```

### Client

The client is a little bit more complex. However, it can be devided in different parts. The main parts are:

* `load_partition()`
    * load the right dataset
* `train()`
    * perfom the local training
* `test()`
    * evaluating the training results
* `CifarClient(fl.client.NumPyClient)`
    * starts Flower client
* `main()`
    * start all previous process in a main() file

The `load_partition()` loads the datasets saved in the `/data` dierctory.  
You can directly see that the training and evaluation process is defined within the client. We are using PyTorch to train and evaluate the model with the parameters given in the chapter [Training Setup](#training-setup). 

The Flower client `CifarClient(fl.client.NumPyClient)` has the usual structure:

* get_paramaters()
* set_parameters()
* fit()
* evaluate()

We will take a closer look at `set_parameters()` in order to demonstrate the difference between FedAvg and FedBN. 

```python 
def set_parameters(self, parameters: List[np.ndarray])-> None:
    self.model.train()
    if self.mode == 'fedbn':
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
    else:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
```

You can see that the local clients take all model parameters and set them for the FedAvg strategy to train a new local model. However, in the case of FedBN, the parameters for the BN layer are excluded. 

