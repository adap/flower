# FedProx MNIST

The following baseline replicates the experiments in *Federated Optimization in Heterogeneous Networks* (Li et al., 2018), which proposed the FedProx algorthim.

**Paper Abstract:** 

<center>
<i>Federated Learning is a distributed learning paradigm with two key challenges that differentiate it from traditional distributed optimization: (1) significant variability in terms of the systems characteristics on each device in the network (systems heterogeneity), and (2) non-identically distributed data across the network (statistical heterogeneity). In this work, we introduce a framework, FedProx, to tackle heterogeneity in federated networks. FedProx can be viewed as a generalization and re-parametrization of FedAvg, the current state-of-the-art method for federated learning. While this re-parameterization makes only minor modifications to the method itself, these modifications have important ramifications both in theory and in practice. Theoretically, we provide convergence guarantees for our framework when learning over data from non-identical distributions (statistical heterogeneity), and while adhering to device-level systems constraints by allowing each participating device to perform a variable amount of work (systems heterogeneity). Practically, we demonstrate that FedProx allows for more robust convergence than FedAvg across a suite of realistic federated datasets. In particular, in highly heterogeneous settings, FedProx demonstrates significantly more stable and accurate convergence behavior relative to FedAvg---improving absolute test accuracy by 22% on average.</i>
</center>

**Paper Authors:** 

Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar and Virginia Smith.

Note: If you use this implementation in your work, please remember to cite the original authors of the paper. 

**[Link to paper.](https://arxiv.org/abs/1812.06127)**

## Training Setup

### CNN Architecture

The CNN architecture is detailed in the paper and used to create the **FedProx MNIST** baseline.

| Layer | Details|
| ----- | ------ |
| 1 | Conv2D(1, 32, 5, 1, 1) <br/> ReLU, MaxPool2D(2, 2, 1)  |
| 2 | Conv2D(32, 64, 5, 1, 1) <br/> ReLU, MaxPool2D(2, 2, 1) |
| 3 | FC(64 * 7 * 7, 512) <br/> ReLU |
| 5 | FC(512, 10) |

### Training Paramaters

| Description | Value |
| ----------- | ----- |
| loss | cross entropy loss |
| optimizer | SGD with proximal term |
| learning rate | 0.03 (by default) |
| local epochs | 5 (by default) |
| local batch size | 10 (by default) |

## Running experiments

The `config.yaml` file containing all the tunable hyperparameters and the necessary variables can be found under the `conf` folder.
[Hydra](https://hydra.cc/docs/tutorials/) is used to manage the different parameters experiments can be ran with. 

To run using the default parameters, just enter `python main.py`, if some parameters need to be overwritten, you can do it like in the following example: 

```sh
python main.py num_epochs=5 num_rounds=1000 iid=True
``` 

Results will be stored as timestamped folders inside either `outputs` or `multiruns`, depending on whether you perform single- or multi-runs. 
