
**Federated Averaging CIFAR10**

The following baseline replicates the experiments in “Communication-Efficient Learning of Deep Networks from Decentralized Data” (McMahan et al., 2017), which was the first paper to coin the term Federated Learning and to propose the FederatedAveraging algorithm.

**Paper Abstract:** 

Modern mobile devices have access to a wealth of data suitable for learning models, which in turn can greatly improve the user experience on the device. For example, language models can improve speech recognition and text entry, and image models can automatically select good photos.

However, this rich data is often privacy sensitive, large in quantity, or both, which may preclude logging to the data center and training there using conventional approaches. We advocate an alternative that leaves the training data distributed on the mobile devices, and learns a shared model by aggregating locally-computed updates. We term this decentralized approach Federated Learning. We present a practical method for the federated learning of deep networks based on iterative model averaging, and conduct an extensive empirical evaluation, considering five different model architectures and four datasets. These experiments demonstrate the approach is robust to the unbalanced and non-IID data distributions that are a defining characteristic of this setting. Communication costs are the principal constraint, and we show a reduction in required communication rounds by 10-100\* — as compared to synchronized stochastic gradient descent

**Paper Authors:** 

H.Brendan McMahan,  Eider Moore, Daniel Ramage, Seth Hampson, and Blaise  Aguera yarcas.

Note: If you use this implementation in your work, please remember to cite the original authors of the paper. 

Link to Paper:   <https://arxiv.org/abs/1602.05629>

##TRAINING SETUP	

##CNN ARCHITECTURE

The CNN architecture is detailed in the paper and used to create the “Federated Averaging CIFAR10” baseline.


|LAYER|`      `DETAILS|
|1|Conv2D(3, 16, 3, 1, 1),  ReLU, MaxPool2D(2, 2, 1)|
|2|conv2D(16, 32, 3, 1, 1),  ReLU, MaxPool2D(2, 2, 1)|
|3|<p>conv2D(32, 64, 3, 1, 1),  ReLU, </p><p>MaxPool2D(2, 2, 1), dropout(0.5)</p>|
|4|FC(64 * 4 * 4, 512), ReLU, dropout(0.5)|
|5|FC(512, 128),  ReLU, dropout(0.5)|
|6|FC(128, 10)|



#TRAINING PARAMETERS

|DESCRIPTION|VALUE|
|loss|cross entropy loss|
|optimizer|SGD|
|Learning rate|0.01(by default)|
|Local epochs|10 (by default)|
|Local batch size|64 (by default)|

#RUNNING EXPERIMENTS

The config.yaml file containing all the tunable hyperparameters and the necessary variables can be found under the conf folder.

[Hydra](https://hydra.cc/docs/tutorials/) is used to manage the different parameters experiments can be run with. 

To run using the default parameters, just enter python main.py. If some parameters need to be overwritten, you can do it like in the following example 


< p bgcolor="light blue "> python main.py num_epochs=10 num_rounds=300 iid=True </p>


Results will be stored as timestamped folders inside either outputs or multiruns, depending on whether you perform single- or multi-runs. 

#OUTPUTS

To help visualize results, the script also plots evaluation curves. Here is an example:

<p align="center”

<img src="docs/results/centralized_loss_iid_balanced_C=100_B=64_E=10_R=400" alt="Centralized loss result" width="400”> 

/>

<p align="center”

<img src="docs/results/centralized_metrics_iid_balanced_C=100_B=64_E=10_R=400" alt="Centralized Accuracy result" width="400”> />

You will also find the saved history in the docs/results/ folder,  here `C` is referring to the number of clients, `B` the batch size, ‘E` the number of local epochs, `R` the number of rounds, and `stag`the proportion of clients that are unreachable at each round.


