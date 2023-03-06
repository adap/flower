# LEAF
The following baseline replicates the experiments conducted in the following paper regarding the FEMNIST dataset.

**Name**: LEAF: A Benchmark for Federated Settings

**Authors**: Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konečný, H. Brendan McMahan, Virginia Smith, Ameet Talwalkar

**Abstract**: Modern federated networks, such as those comprised of wearable devices, mobile phones, or autonomous vehicles, generate massive amounts of data each day. This wealth of data can help to learn models that can improve the user experience on each device. However, the scale and heterogeneity of federated data presents new challenges in research areas such as federated learning, meta-learning, and multi-task learning. As the machine learning community begins to tackle these challenges, we are at a critical time to ensure that developments made in these areas are grounded with realistic benchmarks. To this end, we propose LEAF, a modular benchmarking framework for learning in federated settings. LEAF includes a
suite of open-source federated datasets, a rigorous evaluation framework, and a set of reference implementations, all geared towards capturing the obstacles and intricacies of practical federated environments.

##  FEMNIST
FEMNIST is an extended version of the MNIST dataset for federated learning. Each author of the letters and digits corresponds to a different division of the dataset.

### Conducted Experiments
The conducted experiments were used to evaluate the model towards the following: 
1) System metrics.
2) Performance (using a baseline and an additional pipeline).

#### System metrics
Determining system budget in the total number of FLOPS (across all devices) and bytes written (uploaded to the network) needed to reach the threshold of 75% (per sample) accuracy.

The authors compared the results for minibatch SGD and FedAvg.

The compared hyperparameters in each strategy (see Figure 3 in the paper).

For FedAvg:
* number of clients per round,
* number of epochs each client trained locally.

For minibatch SGD:
* number of clients per round,
* percentage of data used per client.

Global parameters:
* data: authors subsample 5% of the data
* model: Neural Network with two convolutional layers, followed by a Pooling layer, and a Dense layer with 2048 units

Strategy details:
* FedAvg with has a learning rate of 4 * 10^{-3} 
* minibatch SGD  has a learning rate of 6 * 10^{-2}

#### Performance
The authors treat the FedAvg as the baseline and meta-learning method - Reptile as an additional proposed pipeline.

FedAvg - 74.72%. Reptile - 80.24 %.
Global parameters:
* in each device, the data is divided into sets of 60% for training, 20% for validation, and 20% for test,
* model: Neural Network with two convolutional layers, followed by a Pooling layer, and a Dense layer with 2048 units

Strategy details:
* 1 000 rounds, 5 clients per round, local learning rate of 10^{-3},
* minibatch SGD
  * has a size 10
  * is trained for 5 minibatches,
  * is evaluated on an unseen set of test devices
* Reptile
  * has a linearly decaying meta-learning rate that goes from 2 to 0,
  * is evaluated by fine-tuning each test device for 50 mini-batches of size 5

#### Running experiments
Go to the femnist directory and execute the following command.

```bash
python main.py --config-name acc_config; python main.py --config-name modified_acc_config; python main.py --config-name sys_ana_fedavg_e1_c3; python main.py --config-name sys_ana_fedavg_e100_c3; python main.py --config-name sys_ana_fedavg_e1_c35
```

## LEAF Additional Resources

* code https://github.com/TalwalkarLab/leaf
* documentation https://leaf.cmu.edu/build/html/index.html
