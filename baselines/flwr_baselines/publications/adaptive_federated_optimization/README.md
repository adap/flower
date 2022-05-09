# Adaptive Federated Optimization 

The following baseline replicates the experiments in *Adaptive Federated Optimization*, which explores the use of various server-side optimizers for different datasets.

**Paper Abstract:** 

<center>
<i>Federated learning is a distributed machine learning paradigm in which a large
number of clients coordinate with a central server to learn a model without sharing
their own training data. Standard federated optimization methods such as Federated Averaging (FEDAVG) are often difficult to tune and exhibit unfavorable
convergence behavior. In non-federated settings, adaptive optimization methods
have had notable success in combating such issues. In this work, we propose federated versions of adaptive optimizers, including ADAGRAD, ADAM, and YOGI,
and analyze their convergence in the presence of heterogeneous data for general
nonconvex settings. Our results highlight the interplay between client heterogeneity
and communication efficiency. We also perform extensive experiments on these
methods and show that the use of adaptive optimizers can significantly improve the
performance of federated learning.</i>
</center>

**Paper Authors:** 

Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,Jakub Konecný, Sanjiv Kumar, H. Brendan McMahan.


Note: If you use this implementation in your work, please remember to cite the original authors of the paper. 

**[Link to paper.](https://arxiv.org/pdf/2003.00295.pdf)**

## Currently implemented

We have currently implemented the following experiments: 

| Dataset  | Model    | Strategies                          |
| -------- | -------- | ----------------------------------- |
| CIFAR10  | ResNet18 | FeAvg, FedAvgM, FedYogi, FedAdam, FedAdagrad |
| CIFAR100 | ResNet18 | FeAvg, FedAvgM, FedYogi, FedAdam, FedAdagrad |

More experiments to be added soon. 

## Running experiments

Experiments are organized by dataset under the `conf` folder using [hydra](https://hydra.cc/docs/tutorials/), e.g. `conf/cifar10` for the CIFAR10 dataset. 
Each dataset contains a `config.yaml` file containing common setup variables and a `strategy` subfolder with different strategy parameters.

You can run specific experiments by passing a `conf` subfolder and a given `strategy` as follows. 
```sh
python main.py --config-path conf/cifar10 strategy=fedyogi
``` 

Otherwise, you can choose to run multiple configurations sequentially as follows:
```sh
python main.py -m --config-path conf/cifar10 strategy=fedyogi,fedadagrad,fedadam,fedavg
``` 
Results will be stored as timestamped folders inside either `outputs` or `multiruns`, depending on whether you perform single- or multi-runs. 

### Example outputs CIFAR10

To help visualize results, the script also plots evaluation curves. Here are some examples:
<center>
<img src="cifar10_fedavg.jpeg" alt="CIFAR10 - FedAvg" width="400" />
<img src="cifar10_fedadam.jpeg" alt="CIFAR10 - FedAdam" width="400" />
</center>
