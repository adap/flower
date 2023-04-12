# LEAF
The following baseline replicates the experiments conducted in the following paper regarding the FEMNIST dataset.

**Name**: LEAF: A Benchmark for Federated Settings

**Authors**: Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konečný, H. Brendan McMahan, Virginia Smith, Ameet Talwalkar

**Abstract**: Modern federated networks, such as those comprised of wearable devices, mobile phones, or autonomous vehicles, generate massive amounts of data each day. This wealth of data can help to learn models that can improve the user experience on each device. However, the scale and heterogeneity of federated data presents new challenges in research areas such as federated learning, meta-learning, and multi-task learning. As the machine learning community begins to tackle these challenges, we are at a critical time to ensure that developments made in these areas are grounded with realistic benchmarks. To this end, we propose LEAF, a modular benchmarking framework for learning in federated settings. LEAF includes a
suite of open-source federated datasets, a rigorous evaluation framework, and a set of reference implementations, all geared towards capturing the obstacles and intricacies of practical federated environments.

##  FEMNIST
FEMNIST is an extended version of the MNIST dataset for federated learning. The dataset is divided by writer of the handwritten images.

### Experiments
The code reproduces an experiment to evaluate the performance of FedAvg on the FEMNIST dataset(the experiment described in Table 2 in the article).
However, the paper also describes different experiments conducted towards measuring system analysis (computational performance and network communication) but will not be covered in this code.

## Running experiments
Go to the femnist directory and execute the following command. It will download the dataset, preprocess it and start the federated learning.

```bash
python main.py --config-name table2_leaf_paper
```
We also provide an alternative set of parameters (more about it below)
```bash
python main.py --config-name table2_flwr_baseline
```
The parameters are handled by hydra, you can create your own config in the conf directory or overwrite a selected set of them.

## Parameters
### Model
CNN model

| Layer Number | Details                                            |
|--------------|----------------------------------------------------|
| 1            | `nn.Conv2d(1, 32, kernel_size=5, padding="same")`  |
|              | `nn.ReLU()`                                        |
|              | `nn.MaxPool2d(kernel_size=2, stride=2)`            |
| 2            | `nn.Conv2d(32, 64, kernel_size=5, padding="same")` |
|              | `nn.ReLU()`                                        |
|              | `nn.MaxPool2d(kernel_size=2, stride=2)`            |
| 3            | `nn.Linear(7 * 7 * 64, 2048)`                      |
|              | `nn.ReLU()`                                        |
| 4            | `nn.Linear(2048, num_classes)`                     |

### Training
#### General parameters

| Parameter               | Value         |
|-------------------------|---------------|
| strategy                | FedAvg        |
| num_rounds              | 1_000         |
| num_clients_per_round   | 5             |
| loss                    | cross entropy |
| optimizer               | SGD           |
| learning_rate           | 0.001         |
| dataset_fraction        | 0.05          |
| batch_size              | 10            |


#### Differences

| Parameter               | LEAF paper | Flower Baseline |
|-------------------------|------------|-----------------|
| train_time              | 5 batches  | 5 epochs        |
| same_train_test_clients | True       | False           |
| train_fraction          | 0.6        | 0.9             |
| valid_fraction          | 0.2        | 0.0             |
| test_fraction           | 0.2        | 0.1             |

The additional set of parameters was created to mitigate the potential error and inconsistencies between the code and description in the paper.

## Output
The loss and metrics history is saved to the results directory in a CSV format.
Here is an example:

|     | distributed_test_accuracy | distributed_train_loss | distributed_train_accuracy | distributed_test_loss |
|-----|---------------------------|------------------------|----------------------------|-----------------------|
| 0   | 0.009925558312655087      | 0.4103852334664825     | 0.07564790089607601        | 0.4884035895914721    |
| 1   | 0.08898420479302832       | 0.3784615112356397     | 0.07674206155679399        | 0.4523232134144291    |

## Other experiments
Here is a short summary of the parameter needed for experiments from the Figure 3- determining system budget in the total number of FLOPS (across all devices) and bytes written (uploaded to the network) needed to reach the threshold of 75% (per sample) accuracy.

| Parameter                        | 1      | 2      | 3      | 4             | 5             | 6             |
|----------------------------------|--------|--------|--------|---------------|---------------|---------------|
| strategy                         | FedAvg | FedAvg | FedAvg | minibatch SGD | minibatch SGD | minibatch SGD |
| learning_rate                    | 0.004  | 0.004  | 0.004  | 0.06          | 0.06          | 0.06          |
| num_rounds                       | 1_000  | 1_000  | 1_000  | 1_000         | 1_000         | 1_000         |
| num_clients_per_round            | 3      | 3      | 35     | 3             | 3             | 35            |
| epochs_per_round/%of the dataset | 1      | 100    | 1      | 100%          | 10%           | 100%          |

Please note that the paper states that different set of client is used for the training and evalution but the code does the opposite.
The train test split in not explicitly mentionted but the default in the code is 90% train and 10% test division.

## LEAF Additional Resources

* code https://github.com/TalwalkarLab/leaf
* documentation https://leaf.cmu.edu/build/html/index.html
