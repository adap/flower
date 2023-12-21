---
title: HeteroFL - Computation And Communication Efficient Federated Learning For Heterogeneous Clients
url: https://openreview.net/forum?id=TNkPBBYFkXg
labels: [system heterogeneity, image classification]
dataset: [MNIST, CIFAR10]
---

# HeteroFL : Computation And Communication Efficient Federated Learning For Heterogeneous Clients

**Paper:** [openreview.net/forum?id=TNkPBBYFkXg](https://openreview.net/forum?id=TNkPBBYFkXg)

**Authors:** Enmao Diao, Jie Ding, Vahid Tarokh

**Abstract:** Federated Learning (FL) is a method of training machine learning models on private data distributed over a large number of possibly heterogeneous clients such as mobile phones and IoT devices. In this work, we propose a new federated learning framework named HeteroFL to address heterogeneous clients equipped with very different computation and communication capabilities. Our solution can enable the training of heterogeneous local models with varying computation complexities and still produce a single global inference model. For the first time, our method challenges the underlying assumption of existing work that local models have to share the same architecture as the global model. We demonstrate several strategies to enhance FL training and conduct extensive empirical evaluations, including five computation complexity levels of three model architecture on three datasets. We show that adaptively distributing subnetworks according to clients’ capabilities is both computation and communication efficient.


## About this baseline

**What’s implemented:**  The code in this directory is an implementation of HeteroFL in pytorch using flower. The code incorporates references from the authors' implementation. Implementation of custom model split and aggregation as suggested by @negedng, is available [here](https://github.com/msck72/heterofl_custom_aggregation). By modifying the configuration in the base.yaml, the results in the paper can be replicated, with both fixed and dynamic computational complexities among clients.

**Key Terminology:**
+ *Model rate* defines the computational complextiy of a client. Authors have defined five different computation complexity levels {a, b, c, d, e} with the hidden channel shrinkage ratio r = 0.5. 

+ *Model split mode* specifies whether the computaional complexities of clients are fixed (throughout the experiment), or whether they are dynamic (change their mode_rate/computational-complexity every-round). 

+ *Model mode* determines the proportionality of clients with various computation complexity levels, for example, a4-b2-e4 determines at each round, proportion of clients with computational complexity level a = 4 / (4 + 2 + 4) * num_clients , similarly, proportion of clients with computational complexity level b = 2 / (4 + 2 + 4) * num_clients and so on.

**Implementation Insights:**
*ModelRateManager* manages the model rate of client in simulation, which changes the model rate based on the model mode of the setup and *ClientManagerHeterofl* keeps track of model rates of the clients, so configure fit knows which/how-much subset of the model that needs to be sent to the client.

**Datasets:** The code utilized benchmark MNIST and CIFAR-10 datasets from Pytorch's torchvision for its experimentation.

**Hardware Setup:**  The experiments were run on Google colab pro with 50GB RAM and T4 TPU. For MNIST dataset & CNN model, it approximatemy takes 1.5 hours to complete 200 rounds while for CIFAR10 dataset & ResNet18 model it takes around 3-4 hours to complete 400 rounds (may vary based on the model-mode of the setup).

**Contributors:** M S Chaitanya Kumar [(github.com/msck72)](github.com/msck72)


## Experimental Setup

**Task:** Image Classification.
**Model:** This baseline uses two models:
+ Convolutional Neural Network(CNN) model is used for MNIST dataset.
+ PreResNet (preactivated ResNet) model is used for CIFAR10 dataset.

These models use static batch normalization (sBN) and they incorporate a Scaler module following each convolutional layer.

**Dataset:** This baseline includes MNIST and CIFAR10 datasets. 
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

**Training Hyperparameters:** 
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

The hyperparameters of Fedavg baseline are available in [Liang et al (2020)](https://arxiv.org/abs/2001.01523).

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

python -m heterofl.main  # Which runs the heterofl with arguments availbale in heterfl/conf/base.yaml

# We could override the settings that were specified in base.yaml using the command-line-arguments
# Here's an example for changing the dataset name, non-iid and model
python -m heterofl.main dataset.dataset_name='CIFAR10' dataset.iid=False model.model_name='resnet18'

# Similarly, another example for changing num_rounds, model_split_mode, and model_mode
python -m heterofl.main num_rounds=400 control.model_split_mode='dynamic' control.model_mode='a1-b1'

# Similarly, another example for changing num_rounds, model_split_mode, and model_mode
python -m heterofl.main num_rounds=400 control.model_split_mode='dynamic' control.model_mode='a1-b1'

```


## Expected Results

```bash
# running the multirun for IID-MNIST with various model-modes using default config
python -m heterofl.main --multirun control.model_mode='a1','a1-e1','a1-b1-c1-d1-e1'

# running the multirun for IID-CIFAR10 dataset with various model-modes by modifying default config
python -m heterofl.main --multirun control.model_mode='a1','a1-e1','a1-b1-c1-d1-e1' dataset.dataset_name='CIFAR10' model.model_name='resnet18' num_rounds=400 strategy.lr=0.1 strategy.milestones=[150, 250]

# running the multirun for non-IID-MNIST with various model-modes by modifying default config
python -m heterofl.main --multirun control.model_mode='a1','a1-e1','a1-b1-c1-d1-e1' dataset.iid=False num_rounds=400 strategy.milestones=[200]

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
| FedAvg | 1 | 633.226  K | 1.264128  M | 2.416 | 97.85 | 97.76 | 97.74 |


<br>

Results of the combination of various computation complexity levels for **CIFAR10** dataset with **dynamic** scenario(where a client does not belong to a fixed computational complexity level):
> *The HeteroFL paper reports a model with 1.8M parameters for their FedAvg baseline. However, as stated by the paper authors, those results are borrowed from [Liang et al (2020)](https://arxiv.org/abs/2001.01523), which uses a small CNN with fewer parameters (~64K as shown in this table below). We believe the HeteroFL authors made a mistake when reporting the number of parameters. We borrowed the model from Liang et al (2020)'s [repo](https://github.com/pliang279/LG-FedAvg/blob/master/models/Nets.py)*

<table align="center">
<tbody>
	<tr>
		<th rowspan=3>Model</th>
		<th rowspan=3>Ratio</th>
		<th rowspan=3>Parameters</th>
		<th rowspan=3>FLOPS</th>
		<th rowspan=3>Space(MB)</th>
		<th rowspan=3>IID-accuracy</th>
		<th colspan=4> non-IID</th>
	</tr>
	<tr>
		<th colspan=2>local-acc</th>
		<th colspan=2>global-acc</th>
	</tr>
 <tr>
		<th>800th</th>
		<th>Best of 800</th>
		<th>800th</th>
		<th>Best of 800</th>
	</tr>
<tr> <td align="center"> a			</td><td> 1 </td><td> 9622.09  K	</td><td> 330.246 M </td><td> 36.705	</td><td> 90.83 </td><td> 89.04 </td><td> 92.41 </td><td> 48.72 </td><td> 59.29	</td></tr>
<tr> <td align="center"> a-e		</td><td> 0.502 </td><td> 4830.22  K	</td><td> 165.859 M </td><td> 18.426	</td><td> 89.98 </td><td> 87.98 </td><td> 91.25	</td><td> 50.16 </td><td> 57.66	</td></tr>
<tr> <td align="center"> a-b-c-d-e	</td><td> 0.267 </td><td> 2564.95  K	</td><td> 88.354 M	 </td><td> 9.785	</td><td> 87.46 </td><td> 89.75 </td><td> 91.19	</td><td> 46.96 </td><td> 55.6	</td></tr>
<tr> <td align="center"> b			</td><td> 1 </td><td> 2408.746  K	</td><td> 83.334 M	 </td><td> 9.189	</td><td> 88.59 </td><td> 89.31 </td><td> 92.07	</td><td> 49.85 </td><td> 60.79	</td></tr>
<tr> <td align="center"> b-e		</td><td> 0.508 </td><td> 1223.548  K	</td><td> 42.403 M	 </td><td> 4.667	</td><td> 89.23 </td><td> 90.93 </td><td> 92.3	</td><td> 55.46 </td><td> 61.98	</td></tr>
<tr> <td align="center"> b-c-d-e	</td><td> 0.332 </td><td> 800.665  K	</td><td> 27.881 M	 </td><td> 3.054	</td><td> 87.61 </td><td> 89.23 </td><td> 91.83	</td><td> 51.59 </td><td> 59.4	</td></tr> 
<tr> <td align="center"> c			</td><td> 1 </td><td> 603.802  K	</td><td> 21.220 M	 </td><td> 2.303	</td><td> 85.74 </td><td> 89.83 </td><td> 91.75	</td><td> 44.03 </td><td> 58.26	</td></tr>
<tr> <td align="center"> c-e		</td><td> 0.532 </td><td> 321.076  K	</td><td> 11.345 M	 </td><td> 1.225	</td><td> 87.32 </td><td> 89.28 </td><td> 91.56	</td><td> 53.43 </td><td> 59.5	</td></tr> 
<tr> <td align="center"> c-d-e		</td><td> 0.438 </td><td> 264.638  K	</td><td> 9.396 M	 </td><td> 1.010	</td><td> 85.59 </td><td> 91.48 </td><td> 92.05	</td><td> 58.26 </td><td> 61.79	</td></tr>
<tr> <td align="center"> d			</td><td> 1 </td><td> 151.762  K	</td><td> 5.498 M	 </td><td> 0.579	</td><td> 82.91 </td><td> 90.81 </td><td> 91.47	</td><td> 55.95 </td><td> 58.34	</td></tr>
<tr> <td align="center"> d-e		</td><td> 0.626 </td><td> 95.056  K		</td><td> 3.485 M	 </td><td> 0.363	</td><td> 82.77 </td><td> 88.79 </td><td> 90.13	</td><td> 48.49 </td><td> 54.18	</td></tr>
<tr> <td align="center"> e			</td><td> 1 </td><td> 38.35  K		</td><td> 1.471 M	 </td><td> 0.146	</td><td> 76.53 </td><td> 90.05 </td><td> 90.91	</td><td> 54.68 </td><td> 57.05	</td></tr>
<tr>
	<th colspan=6 align="center">FedAvg</th>
	<th><sub>1800th</sub></th>
	<th><sub>Best of 1800</sub></th>
	<th><sub>1800th</sub></th>
	<th><sub>Best of 1800</sub></th>
</tr>
<tr> <td align="center"> FedAvg			</td><td> 1 </td><td> 64.102  K		</td><td> 1.3202  M	 </td><td> 0.2446	</td><td>70.65</td><td> 56.69 </td><td> 58.72	</td><td> 56.76 </td><td> 58.64 </td></tr>
</tbody>

</table>



