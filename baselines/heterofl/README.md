---
title: "HeteroFL: Computation And Communication Efficient Federated Learning For Heterogeneous Clients"
url: https://openreview.net/forum?id=TNkPBBYFkXg
labels: [system heterogeneity, image classification]
dataset: [MNIST, CIFAR-10]
---

# HeteroFL: Computation And Communication Efficient Federated Learning For Heterogeneous Clients

**Paper:** [openreview.net/forum?id=TNkPBBYFkXg](https://openreview.net/forum?id=TNkPBBYFkXg)

**Authors:** Enmao Diao, Jie Ding, Vahid Tarokh

**Abstract:** Federated Learning (FL) is a method of training machine learning models on private data distributed over a large number of possibly heterogeneous clients such as mobile phones and IoT devices. In this work, we propose a new federated learning framework named HeteroFL to address heterogeneous clients equipped with very different computation and communication capabilities. Our solution can enable the training of heterogeneous local models with varying computation complexities and still produce a single global inference model. For the first time, our method challenges the underlying assumption of existing work that local models have to share the same architecture as the global model. We demonstrate several strategies to enhance FL training and conduct extensive empirical evaluations, including five computation complexity levels of three model architecture on three datasets. We show that adaptively distributing subnetworks according to clients’ capabilities is both computation and communication efficient.


## About this baseline

**What’s implemented:**  The code in this directory is an implementation of HeteroFL in PyTorch using Flower. The code incorporates references from the authors' implementation. Implementation of custom model split and aggregation as suggested by [@negedng](https://github.com/negedng), is available [here](https://github.com/msck72/heterofl_custom_aggregation). By modifying the configuration in the `base.yaml`, the results in the paper can be replicated, with both fixed and dynamic computational complexities among clients.

**Key Terminology:**
+ *Model rate* defines the computational complexity of a client. Authors have defined five different computation complexity levels {a, b, c, d, e} with the hidden channel shrinkage ratio r = 0.5. 

+ *Model split mode* specifies whether the computational complexities of clients are fixed (throughout the experiment), or whether they are dynamic (change their mode_rate/computational-complexity every-round). 

+ *Model mode* determines the proportionality of clients with various computation complexity levels, for example, a4-b2-e4 determines at each round, proportion of clients with computational complexity level a = 4 / (4 + 2 + 4) * num_clients, similarly, proportion of clients with computational complexity level b = 2 / (4 + 2 + 4) * num_clients and so on.

**Implementation Insights:**
*ModelRateManager* manages the model rate of client in simulation, which changes the model rate based on the model mode of the setup and *ClientManagerHeterofl* keeps track of model rates of the clients, so configure fit knows which/how-much subset of the model that needs to be sent to the client.

**Datasets:** The code utilized benchmark MNIST and CIFAR-10 datasets from Pytorch's torchvision for its experimentation.

**Hardware Setup:**  The experiments were run on Google colab pro with 50GB RAM and T4 TPU. For MNIST dataset & CNN model, it approximately takes 1.5 hours to complete 200 rounds while for CIFAR10 dataset & ResNet18 model it takes around 3-4 hours to complete 400 rounds (may vary based on the model-mode of the setup).

**Contributors:** M S Chaitanya Kumar [(github.com/msck72)](https://github.com/msck72)


## Experimental Setup

**Task:** Image Classification.
**Model:** This baseline uses two models:
+ Convolutional Neural Network(CNN) model is used for MNIST dataset.
+ PreResNet (preactivated ResNet) model is used for CIFAR10 dataset.

These models use static batch normalization (sBN) and they incorporate a Scaler module following each convolutional layer.

**Dataset:** This baseline includes MNIST and CIFAR10 datasets. 

| Dataset | #Classes | IID Partition | non-IID Partition |
| :---: | :---: | :---: | :---: | 
| MNIST<br>CIFAR10 |  10| Distribution of equal number of data examples among n clients | Distribution of data examples such that each client has at most 2 (customizable) classes |


**Training Hyperparameters:** 

| Description | Data Setting | MNIST | CIFAR-10 |
| :---: | :---: |  :---:| :---: |
Total Clients  | both | 100 | 100 |
Clients Per Round | both | 100 | 100
Local Epcohs | both | 5 | 5
Num. ROunds | IID <br> non-IID| 200<br>400 | 400<br>800
Optimizer | both | SGD | SGD
Momentum | both | 0.9 | 0.9
Weight-decay | both | 5.00e-04 | 5.00e-04
Learning Rate | both | 0.01 | 0.1
Decay Schedule | IID <br> non-IID| [100]<br>[150, 250] | [200]<br>[300,500]
Hidden Layers | both | [64 , 128 , 256 , 512] | [64 , 128 , 256 , 512]


The hyperparameters of Fedavg baseline are available in [Liang et al (2020)](https://arxiv.org/abs/2001.01523).

## Environment Setup

To construct the Python environment, simply run:

```bash
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
To run HeteroFL experiments in poetry activated environment:
```bash  
# The main experiment implemented in your baseline using default hyperparameters (that should be setup in the Hydra configs)
# should run (including dataset download and necessary partitioning) by executing the command:

python -m heterofl.main  # Which runs the heterofl with arguments available in heterfl/conf/base.yaml

# We could override the settings that were specified in base.yaml using the command-line-arguments
# Here's an example for changing the dataset name, non-iid and model
python -m heterofl.main dataset.dataset_name='CIFAR10' dataset.iid=False model.model_name='resnet18'

# Similarly, another example for changing num_rounds, model_split_mode, and model_mode
python -m heterofl.main num_rounds=400 control.model_split_mode='dynamic' control.model_mode='a1-b1'

# Similarly, another example for changing num_rounds, model_split_mode, and model_mode
python -m heterofl.main num_rounds=400 control.model_split_mode='dynamic' control.model_mode='a1-b1'

```
To run FedAvg experiments:
```bash
python -m heterofl.main --config-name fedavg
# Similarly to the commands illustrated above, we can modify the default settings in the fedavg.yaml file.
```

## Expected Results

```bash
# running the multirun for IID-MNIST with various model-modes using default config
python -m heterofl.main --multirun control.model_mode='a1','a1-e1','a1-b1-c1-d1-e1'

# running the multirun for IID-CIFAR10 dataset with various model-modes by modifying default config
python -m heterofl.main --multirun control.model_mode='a1','a1-e1','a1-b1-c1-d1-e1' dataset.dataset_name='CIFAR10' model.model_name='resnet18' num_rounds=400 optim_scheduler.lr=0.1 strategy.milestones=[150, 250]

# running the multirun for non-IID-MNIST with various model-modes by modifying default config
python -m heterofl.main --multirun control.model_mode='a1','a1-e1','a1-b1-c1-d1-e1' dataset.iid=False num_rounds=400 optim_scheduler.milestones=[200]

# similarly, we can perform for various model-modes, datasets. But we cannot multirun with both non-iid and iid at once for reproducing the tables below, since the number of rounds and milestones for MultiStepLR are different for non-iid and iid. The tables below are the reproduced results of various multiruns.

#To reproduce the fedavg results
#for MNIST dataset
python -m heterofl.main --config-name fedavg --multirun dataset.iid=True,False
# for CIFAR10 dataset
python -m heterofl.main --config-name fedavg --multirun num_rounds=1800 dataset.dataset_name='CIFAR10' dataset.iid=True,False dataset.batch_size.train=50 dataset.batch_size.test=128 model.model_name='CNNCifar' optim_scheduler.lr=0.1
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
| FedAvg | 1 | 633.226  K | 1.264128  M | 2.416 | 97.85 | 97.76 | 97.74 |


<br>

Results of the combination of various computation complexity levels for **CIFAR10** dataset with **dynamic** scenario(where a client does not belong to a fixed computational complexity level):
> *The HeteroFL paper reports a model with 1.8M parameters for their FedAvg baseline. However, as stated by the paper authors, those results are borrowed from [Liang et al (2020)](https://arxiv.org/abs/2001.01523), which uses a small CNN with fewer parameters (~64K as shown in this table below). We believe the HeteroFL authors made a mistake when reporting the number of parameters. We borrowed the model from Liang et al (2020)'s [repo](https://github.com/pliang279/LG-FedAvg/blob/master/models/Nets.py). As in the paper, FedAvg was run for 1800 rounds.*


| Model | Ratio | Parameters | FLOPS | Space(MB) | IID-acc | non-IId  local-acc <br> Final &#8195;&#8195;Best| non-IID  global-acc <br> Final &#8195;&#8195; Best|
| :--: | :----: | :-----: | :-------: | :-------: | :----------: | :-----: | :------: |
 a		| 1     | 9622  K	| 330.2 M  	 | 36.705	| 90.83 | 89.04   &#8195;&#8195;  92.41 | 48.72  &#8195;&#8195;  59.29	|
 a-e		| 0.502 | 4830  K	| 165.9 M  	 | 18.426	| 89.98 | 87.98   &#8195;&#8195;  91.25	| 50.16  &#8195;&#8195;  57.66	|
 a-b-c-d-e	| 0.267 | 2565  K	| 88.4 M	 | 9.785	| 87.46 | 89.75   &#8195;&#8195;  91.19	| 46.96  &#8195;&#8195;  55.6	|
 b		| 1     | 2409  K	| 83.3 M	 | 9.189	| 88.59 | 89.31   &#8195;&#8195;  92.07	| 49.85  &#8195;&#8195;  60.79	|
 b-e		| 0.508 | 1224  K	| 42.4 M	 | 4.667	| 89.23 | 90.93   &#8195;&#8195;  92.3	| 55.46  &#8195;&#8195;  61.98	|
 b-c-d-e	| 0.332 | 801  K	| 27.9 M	 | 3.054	| 87.61 | 89.23   &#8195;&#8195;  91.83	| 51.59  &#8195;&#8195;  59.4	| 
 c		| 1     | 604  K	| 21.2 M	 | 2.303	| 85.74 | 89.83   &#8195;&#8195;  91.75	| 44.03  &#8195;&#8195;  58.26	|
 c-e		| 0.532 | 321  K	| 11.4 M	 | 1.225	| 87.32 | 89.28   &#8195;&#8195;  91.56	| 53.43  &#8195;&#8195;  59.5	| 
 c-d-e		| 0.438 | 265  K	| 9.4 M	 | 1.010	| 85.59 | 91.48   &#8195;&#8195;  92.05	| 58.26  &#8195;&#8195;  61.79	|
 d		| 1     | 152  K	| 5.5 M	 | 0.579	| 82.91 | 90.81   &#8195;&#8195;  91.47	| 55.95  &#8195;&#8195;  58.34	|
 d-e		| 0.626 | 95  K	| 3.5 M	 | 0.363	| 82.77 | 88.79   &#8195;&#8195;  90.13	| 48.49  &#8195;&#8195;  54.18	|
 e		| 1     | 38  K	| 1.5 M	 | 0.146	| 76.53 | 90.05   &#8195;&#8195;  90.91	| 54.68  &#8195;&#8195;  57.05	|
|FedAvg	| 1 | 64  K| 1.3  M | 0.2446 | 70.65 | 53.12 &#8195;&#8195; 58.6 | 52.93 &#8195;&#8195; 58.47 |


<!--
| Model | Ratio | Parameters | FLOPS | Space(MB) | IID-accuracy | non-IId-|-local-acc | non-IID-|-global-acc | 
| :--: | :----: | :-----: | :-------: | :-------: | :----------: | :-----: | :------: | :-----: | :-----: |
|  |  |  |  |  |  | **800th** | **Best of 800**|  **800th** | **Best of 800**|
 a		| 1     | 9622.09  K	| 330.246 M  	 | 36.705	| 90.83 | 89.04   |  92.41 	| 48.72  |  59.29	|
 a-e		| 0.502 | 4830.22  K	| 165.859 M  	 | 18.426	| 89.98 | 87.98   |  91.25	| 50.16  |  57.66	|
 a-b-c-d-e	| 0.267 | 2564.95  K	| 88.354 M	 | 9.785	| 87.46 | 89.75   |  91.19	| 46.96  |  55.6	|
 b		| 1     | 2408.746  K	| 83.334 M	 | 9.189	| 88.59 | 89.31   |  92.07	| 49.85  |  60.79	|
 b-e		| 0.508 | 1223.548  K	| 42.403 M	 | 4.667	| 89.23 | 90.93   |  92.3	| 55.46  |  61.98	|
 b-c-d-e	| 0.332 | 800.665  K	| 27.881 M	 | 3.054	| 87.61 | 89.23   |  91.83	| 51.59  |  59.4	| 
 c		| 1     | 603.802  K	| 21.220 M	 | 2.303	| 85.74 | 89.83   |  91.75	| 44.03  |  58.26	|
 c-e		| 0.532 | 321.076  K	| 11.345 M	 | 1.225	| 87.32 | 89.28   |  91.56	| 53.43  |  59.5	| 
 c-d-e		| 0.438 | 264.638  K	| 9.396 M	 | 1.010	| 85.59 | 91.48   |  92.05	| 58.26  |  61.79	|
 d		| 1     | 151.762  K	| 5.498 M	 | 0.579	| 82.91 | 90.81   |  91.47	| 55.95  |  58.34	|
 d-e		| 0.626 | 95.056  K	| 3.485 M	 | 0.363	| 82.77 | 88.79   |  90.13	| 48.49  |  54.18	|
 e		| 1     | 38.35  K	| 1.471 M	 | 0.146	| 76.53 | 90.05   |  90.91	| 54.68  |  57.05	|
| 		| 	|		|**FedAvg** |		|	|<sub>**1800th**</sub>|<sub>**Best of 1800**</sub>|<sub>**1800th**</sub>|<sub>**Best of 1800**</sub>|
|FedAvg		| 1 	| 64.102  K	| 1.3202  M 	| 0.2446 	| 70.65 | 56.69   | 58.72 	| 56.76  | 58.64 	|

-->
