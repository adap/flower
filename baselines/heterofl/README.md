---
title: HeteroFL - Computation And Communication Efficient Federated Learning For Heterogeneous Clients
url: https://openreview.net/forum?id=TNkPBBYFkXg
labels: [system heterogeneity, image classification]
dataset: [MNIST, CIFAR10]
---

# HeteroFL : Computation And Communication Efficient Federated Learning For Heterogeneous Clients

****Paper:**** [openreview.net/forum?id=TNkPBBYFkXg](https://openreview.net/forum?id=TNkPBBYFkXg)

****Authors:**** Enmao Diao, Jie Ding, Vahid Tarokh

****Abstract:**** Federated Learning (FL) is a method of training machine learning models on private data distributed over a large number of possibly heterogeneous clients such as mobile phones and IoT devices. In this work, we propose a new federated learning framework named HeteroFL to address heterogeneous clients equipped with very different computation and communication capabilities. Our solution can enable the training of heterogeneous local models with varying computation complexities and still produce a single global inference model. For the first time, our method challenges the underlying assumption of existing work that local models have to share the same architecture as the global model. We demonstrate several strategies to enhance FL training and conduct extensive empirical evaluations, including five computation complexity levels of three model architecture on three datasets. We show that adaptively distributing subnetworks according to clients’ capabilities is both computation and communication efficient.


## About this baseline

****What’s implemented:****  The code in this directory is an implementation of HeteroFL in pytorch using flower. The code incorporates references from the authors' implementation. Implementation of custom model split and aggregation as suggested by @negedng, is available [here](https://github.com/msck72/heterofl_custom_aggregation). By modifying the configuration in the base.yaml, the results in the paper can be replicated, with both fixed and dynamic computational complexities among clients.

****Key Terminology:****
+ *Model rate* defines the computational complextiy of a client. Authors have defined five different computation complexity levels {a, b, c, d, e} with the hidden channel shrinkage ratio r = 0.5. 

+ *Model split mode* specifies whether the computaional complexities of clients are fixed (throughout the experiment), or whether they are dynamic (change their mode_rate/computational-complexity every-round). 

+ *Model mode* determines the proportionality of clients with various computation complexity levels, for example, a4-b2-e4 determines at each round, proportion of clients with computational complexity level a = 4 / (4 + 2 + 4) * num_clients , similarly, proportion of clients with computational complexity level b = 2 / (4 + 2 + 4) * num_clients and so on.

****Implementation Insights:****
*ModelRateManager* manages the model rate of client in simulation, which changes the model rate based on the model mode of the setup and *ClientManagerHeterofl* keeps track of model rates of the clients, so configure fit knows which/how-much subset of the model that needs to be sent to the client.

****Datasets:**** The code utilized benchmark MNIST and CIFAR-10 datasets from Pytorch's torchvision for its experimentation.

****Hardware Setup:****  The experiments were run on Google colab pro with 50GB RAM and T4 TPU. For MNIST dataset & CNN model, it approximatemy takes 1.5 hours to complete 200 rounds while for CIFAR10 dataset & ResNet18 model it takes around 3-4 hours to complete 400 rounds (may vary based on the model-mode of the setup).

****Contributors:**** M S Chaitanya Kumar [(github.com/msck72)](github.com/msck72)


## Experimental Setup

****Task:**** Image Classification.
****Model:**** This baseline uses two models:
+ Convolutional Neural Network(CNN) model is used for MNIST dataset.
+ PreResNet (preactivated ResNet) model is used for CIFAR10 dataset.

These models use static batch normalization (sBN) and they incorporate a Scaler module following each convolutional layer.

****Dataset:**** This baseline includes MNIST and CIFAR10 datasets. 
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>#classes</th>
      <th>IID partition</th>
      <th>non-IID partition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MNIST</td>
      <td>10</td>
      <td rowspan="2">Distribution of equal number of data examples among n clients</td>
      <td rowspan="2">Distribution of data examples such that each client has at most 2 (customizable) classes</td>
    </tr>
    <tr>
      <td>CIFAR10</td>
      <td>10</td>
    </tr>
  </tbody>
</table>

****Training Hyperparameters:**** 
<table>
  <thead>
    <tr>
      <th colspan="2">Description</th>
      <th>MNIST</th>
      <th>CIFAR10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2">total clients</td>
      <td colspan="2">100</td>
    </tr>
    <tr>
      <td colspan="2">clients per round</td>
      <td colspan="2">100</td>
    </tr>
    <tr>
      <td colspan="2">#local epochs</td>
      <td colspan="2">5</td>
    </tr>
    <tr>
      <td rowspan="2">number of rounds</td>
      <td>IID</td>
      <td>200</td>
      <td>400</td>
    </tr>
    <tr>
      <td>non-IID</td>
      <td>400</td>
      <td>800</td>
    </tr>
    <tr>
      <td colspan="2">optimizer</td>
      <td colspan="2">SGD</td>
    </tr>
    <tr>
      <td colspan="2">momentum</td>
      <td colspan="2">5.00e-04</td>
    </tr>
    <tr>
      <td colspan="2">weight-decay</td>
      <td colspan="2">0.9</td>
    </tr>
    <tr>
      <td colspan="2">learning rate</td>
      <td>0.01</td>
      <td>0.1</td>
    </tr>
        <tr>
      <td rowspan="2">decay schedule</td>
      <td>IID</td>
      <td>[100]</td>
      <td>[200]</td>
    </tr>
    <tr>
      <td>non-IID</td>
      <td>[150, 250]</td>
      <td>[300, 500]</td>
    </tr>
    <tr>
      <td colspan="2">hidden layers</td>
      <td colspan="2">[64 , 128 , 256 , 512]</td>
    </tr>
  </tbody>
</table>



## Environment Setup

```
# Set python version
pyenv install 3.10.6
pyenv local 3.10.6

# Tell poetry to use python 3.10
poetry env use 3.10.6

# install the base Poetry environment
poetry install

# activate the environment
poetry shell
```


## Running the Experiments

```bash  
# The main experiment implemented in your baseline using default hyperparameters (that should be setup in the Hydra configs)
# should run (including dataset download and necessary partitioning) by executing the command:

poetry run -m heterofl.main  # Which runs the heterofl with arguments availbale in heterfl/conf/base.yaml

# We could override the settings that were specified in base.yaml using the command-line-arguments
# Here's an example for changing the dataset name, non-iid and model
poetry run -m heterofl.main dataset.dataset_name='CIFAR10' dataset.iid=False model.model_name='resnet18'

# Similarly, another example for changing num_rounds, model_split_mode, and model_mode
poetry run -m heterofl.main num_rounds=400 control.model_split_mode='dynamic' control.model_mode='a1-b1'

# Similarly, another example for changing num_rounds, model_split_mode, and model_mode
poetry run -m heterofl.main num_rounds=400 control.model_split_mode='dynamic' control.model_mode='a1-b1'

```


## Expected Results

```bash
# running the multirun for IID-MNIST with various model-modes using default config
poetry run -m heterofl.main --multirun control.model_mode='a1','a1-e1','a1-b1-c1-d1-e1'

# running the multirun for IID-CIFAR10 dataset with various model-modes by modifying default config
poetry run -m heterofl.main --multirun control.model_mode='a1','a1-e1','a1-b1-c1-d1-e1' dataset.dataset_name='CIFAR10' model.model_name='resnet18' num_rounds=400 strategy.lr=0.1 strategy.milestones=[150, 250]

# running the multirun for non-IID-MNIST with various model-modes by modifying default config
poetry run -m heterofl.main --multirun control.model_mode='a1','a1-e1','a1-b1-c1-d1-e1' dataset.iid=False num_rounds=400 strategy.milestones=[200]

# similarly, we can perform for various model-modes, datasets. But we cannot multirun with both non-iid and iid at once for reproducing the tables below, since the number of rounds and milestones for MultiStepLR are different for non-iid and iid. The tables below are the reproduced results of various multiruns.
```
<br>
 
Results of the combination of various computation complexity levels for **MNIST** dataset with **dynamic** scenario(where a client does not belong to a fixed computational complexity level):

| Model | Ratio | Parameters | FLOPS | Space(MB) | IID-accuracy | non-IId local-acc | non-IID global-acc |
| :--: | :----: | :-----: | :-------: | :-------: | :----------: | :---------------: | :----------------: |
| a | 1  | 1556.874  K | 80.504  M | 5.939 | 99.47 | 99.82 | 98.87 |
| a-e | 0.502 | 781.734  K | 40.452  M | 2.982 | 99.49 | 99.86 | 98.9 |
| a-b-c-d-e | 0.267 | 415.807  K | 21.625  M | 1.586 | 99.23 | 99.84 | 98.5 |
| b | 1  | 391.37  K | 20.493  M | 1.493 | 99.54 | 99.81 | 98.81 |
| b-e | 0.508  | 198.982  K | 10.447  M | 0.759 | 99.48 | 99.87 | 98.98 |
| b-c-d-e | 0.334  | 130.54  K | 6.905  M | 0.498 | 99.34 | 99.81 | 98.73 |
| c | 1 | 98.922  K | 5.307  M | 0.377 | 99.37 | 99.64 | 97.14 |
| c-e |  0.628  | 62.098  K | 3.363  M | 0.237 | 99.16 | 99.72 | 97.68 |
| c-d-e | 0.441  | 43.5965  K | 2.375  M | 0.166 | 99.28 | 99.69 | 97.27 |
| d | 1 | 25.274  K | 1.418 M | 0.096 | 99.07 | 99.77 | 97.58 |
| d-e | 0.63 | 15.934  K | 0.909  M | 0.0608 | 99.12 | 99.65 | 97.33 |
| e | 1 | 6.594  K | 0.4005  M | 0.025 | 98.46 | 99.53 | 96.5 |

<br>

Results of the combination of various computation complexity levels for **CIFAR10** dataset with **dynamic** scenario(where a client does not belong to a fixed computational complexity level):

| Model | Ratio | Parameters | FLOPS | Space(MB) | IID-accuracy | non-IId local-acc | non-IID global-acc |
| :--: | :-----: | :-----: | :-------: | :-------: | :----------: | :---------------: | :----------------: |
| a | 1  | 11172.17  K |	557.656  M	| 42.618 | 90.83 | 89.45 | 53.59 |
| a-e | 0.502  | 5608.34  K | 280.035  M | 21.394 | 89.98 | 90.72 | 54.67 |
| a-b-c-d-e | 0.267  | 2978.118  K	| 149.048  M |	11.361 | 87.46 | 87.11 | 45.08 |
| b | 1  | 2796.714  K	| 140.416  M | 10.669 | 88.59 | 92.18 | 52.83 |
| b-e | 0.508  | 1420.612  K |	71.415  M | 5.419 | 89.23 | 87.36 | 49.58 |
| b-c-d-e | 0.332  | 929.605  K | 46.896  M | 3.546 | 87.61 | 88.84 | 48.07 |
| c | 1  | 701.018  K | 35.605  M | 2.674 | 85.74 | 88.1 | 54.43 |
| c-e | 0.626  | 438.598  K | 22.378  M | 1.673 | 87.32 | 90.8 | 56.47 |
| c-d-e | 0.438  | 307.2354  K |	15.723  M | 1.172 | 85.59 | 89.63 | 54.26 |
| d | 1  | 176.178  K | 9.152  M | 0.672 | 82.91 | 85.83 | 47.5 |
| d-e | 0.626  | 110.344  K | 5.782  M |	0.421 | 82.77 | 90.38 | 56.24 |
| e | 1  | 44.51  K | 2.413  M | 0.170 | 76.53 | 83.55 | 48.29 |



