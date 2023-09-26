---
title: "Federated Learning on Non-IID Data Silos: An Experimental Study"
url: https://arxiv.org/abs/2102.02079
labels: [data heterogeneity, image classification, benchmark]
dataset: [cifar10, mnist, fashion-mnist]
---

# Federated Learning on Non-IID Data Silos: An Experimental Study

****Paper:**** : [https://arxiv.org/abs/2102.02079](https://arxiv.org/abs/2102.02079)

****Authors:**** : Qinbin Li, Yiqun Diao, Quan Chen, Bingsheng He

****Abstract:**** : Due to the increasing privacy concerns and data regulations, training data have been increasingly fragmented, forming distributed databases of multiple "data silos" (e.g., within different organizations and countries). To develop effective machine learning services, there is a must to exploit data from such distributed databases without exchanging the raw data. Recently, federated learning (FL) has been a solution with growing interests, which enables multiple parties to collaboratively train a machine learning model without exchanging their local data. A key and common challenge on distributed databases is the heterogeneity of the data distribution among the parties. The data of different parties are usually non-independently and identically distributed (i.e., non-IID). There have been many FL algorithms to address the learning effectiveness under non-IID data settings. However, there lacks an experimental study on systematically understanding their advantages and disadvantages, as previous studies have very rigid data partitioning strategies among parties, which are hardly representative and thorough. In this paper, to help researchers better understand and study the non-IID data setting in federated learning, we propose comprehensive data partitioning strategies to cover the typical non-IID data cases. Moreover, we conduct extensive experiments to evaluate state-of-the-art FL algorithms. We find that non-IID does bring significant challenges in learning accuracy of FL algorithms, and none of the existing state-of-the-art FL algorithms outperforms others in all cases. Our experiments provide insights for future studies of addressing the challenges in "data silos".


## About this baseline

****What’s implemented:**** :warning: *_Concisely describe what experiment(s) in the publication can be replicated by running the code. Please only use a few sentences. Start with: “The code in this directory …”_*

****Datasets:**** :warning: *_List the datasets you used (if you used a medium to large dataset, >10GB please also include the sizes of the dataset)._*

****Hardware Setup:**** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

****Contributors:**** :warning: *_let the world know who contributed to this baseline. This could be either your name, your name and affiliation at the time, or your GitHub profile name if you prefer. If multiple contributors signed up for this baseline, please list yourself and your colleagues_*


## Experimental Setup

****Task:**** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

****Model:**** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

****Dataset:**** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

****Training Hyperparameters:**** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

```bash
# Setup the base poetry enviroment from the niid_bench directory
poetry install

# Start the shell
poetry shell

#  Install tqdm, torch, and torchvision using pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tqdm
```


## Running the Experiments
You can run four algorithms fedavg, scaffold, fedprox, and fednova. To run any of them, use any of the corresponding main files. For instance, the following command will run with the default config provided in the corresponding configuration files.

```bash
python -m niid_bench.main_fedprox
```

To change the configuration such as dataset or hyperparameters, specify them as part of the command line arguments.

```bash
python -m niid_bench.main_fedprox dataset_name=mnist partitioning=iid model_t=niid_bench.models.CNNMnist model.input_dim=256
```

The full list of possible arguments and their values are as follows.




## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
