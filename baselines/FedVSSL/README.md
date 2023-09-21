---
title: Federated Self-supervised Learning for Video Understanding
url: https://arxiv.org/abs/2207.01975
labels: [cross-device, videossl] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [UCF-101] # list of datasets you include in your baseline
---

# Federated Self-supervised Learning for Video Understanding
> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.


**paper:** [https://arxiv.org/abs/2207.01975](https://arxiv.org/abs/2207.01975)


**Authors:** Yasar Abbas Ur Rehman, Yan Gao, Jiajun Shen, Pedro Porto Buarque de Gusmao, Nicholas Lane


**Abstract:** The ubiquity of camera-enabled mobile devices has lead to large amounts of unlabelled video data being produced at the edge. Although various self-supervised learning (SSL) methods have been proposed to harvest their latent spatio-temporal representations for task-specific training, practical challenges including privacy concerns and communication costs prevent SSL from being deployed at large scales. To mitigate these issues, we propose the use of Federated Learning (FL) to the task of video SSL. In this work, we evaluate the performance of current state-of-the-art (SOTA) video-SSL techniques and identify their shortcomings when integrated into the large-scale FL setting simulated with kinetics-400 dataset. We follow by proposing a novel federated SSL framework for video, dubbed FedVSSL, that integrates different aggregation strategies and partial weight updating. Extensive experiments demonstrate the effectiveness and significance of FedVSSL as it outperforms the centralized SOTA for the downstream retrieval task by 6.66% on UCF-101 and 5.13% on HMDB-51. 

![alt text](./FVSSL.png)

## About this baseline

**Whats's implemented:** The code in this directory replicates the experiments in * Federated Self-supervised Learning for Video Understanding* (Rehman et al., 2022) for UCF-101, which proposed the FedVSSL algorithm. Specifically, it replicates the results for UCF-101 in Table 4 in the paper.
As common SSL training pipline, this code has two parts: SSL pre-training in FL and downstream fine-tuning.

**Dataset:** UCF-101

**Hardware Setup:** These experiments were on a server with 6 GTX-3090 GPU and 128 CPU threads. 

**Contributers:** Yasar Abbas Ur Rehman and Yan Gao

## Experimental Setup

**Task:** Action Recognition

**Model:** 
* This directory first pretrain Catch-the-Patch (CtP) model from the CtP (see `CtP/pyvrl/models/pretraining/ctp`) repository during FL pretrainin stage The backbone model is R3D-18 (see `/CtP/pyvrl/models/backbones/r3d.py`). 
* After pretraining it finetunes the R3D-18 model on UCF-101 dataset.

**Dataset:** This baselines only include pre-training and fine-tuning with UCF-101 dataset. However, we also provide the script files to generate the partitions for Kinetics-400 datasets. 
For UCF-101 dataset, one can simply run the `dataset_preparation.py` file to download and generate the iid splits for UCF-101 datasets.

| Dataset | #classes | #partitions | partitioning method |  partition settings  |
|:--------|:--------:|:-----------:| :---: |:--------------------:|
| UCF101  |   101    |     10      | randomly partitioned |       uniform        |
| Kinectics-400    |   400    |     100     | randomly partitioned | 8 classes per client |

**Training Hyperparameters:** The following table shows the main hyperparameters for this baseline with their default value (i.e. the value used if you run python main.py directly)

| Description        |            Default Value            |
|:-------------------|:-----------------------------------:|
| total clients      |                 10                  |
| clients per round  |                 10                  | 
| number of rounds	  |                 20                  | 
| client resources	  | {'num_cpus': 2.0, 'num_gpus': 1.0 } | 
| optimizer	  |                 SGD                  | 
| alpha coefficient	 |                 0.9                 | 
| beta coefficient	  |                  1                  | 

## Environment Setup
To construct the Python environment follow these steps:

```bash
# install the base Poetry environment
poetry install

# activate the environment
poetry shell

# install PyTorch with GPU support.
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# install mmcv package
pip install mmcv==1.2.4
```

## Running the Experiments
To run this FedVSSL with UCF-101 baseline, first ensure you have activated your Poetry environment (execute `poetry shell` from this directory), then:

```bash
# run federated SSL training with FedVSSL
python -m FedVSSL.main --pre_train=True # this will run using the default settings.

# you can override settings directly from the command line
python -m fedprox.main mu=1 num_rounds=200 # will set proximal mu to 1 and the number of rounds to 200

# run downstream fine-tuning with pre-trained SSL model
python -m FedVSSL.main --pre_train=False # this will run using the default settings.

```

To run using FedAvg:
```bash
# this will use a variation of FedAvg that drops the clients that were flagged as stragglers
# This is done so to match the experimental setup in the FedProx paper
python -m FedVSSL.main --config-name fedavg

# this config can also be overriden from the CLI
```





# :warning:*_Title of your baseline_*

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

> :warning: This is the template to follow when creating a new Flower Baseline. Please follow the instructions in `EXTENDED_README.md`

> :warning: Please follow the instructions carefully. You can see the [FedProx-MNIST baseline](https://github.com/adap/flower/tree/main/baselines/fedprox) as an example of a baseline that followed this guide.

> :warning: Please complete the metadata section at the very top of this README. This generates a table at the top of the file that will facilitate indexing baselines.

****Paper:**** :warning: *_add the URL of the paper page (not to the .pdf). For instance if you link a paper on ArXiv, add here the URL to the abstract page (e.g. https://arxiv.org/abs/1512.03385). If your paper is in from a journal or conference proceedings, please follow the same logic._*

****Authors:**** :warning: *_list authors of the paper_*

****Abstract:**** :warning: *_add here the abstract of the paper you are implementing_*


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

:warning: _The Python environment for all baselines should follow these guidelines in the `EXTENDED_README`. Specify the steps to create and activate your environment. If there are any external system-wide requirements, please include instructions for them too. These instructions should be comprehensive enough so anyone can run them (if non standard, describe them step-by-step)._


## Running the Experiments

:warning: _Provide instructions on the steps to follow to run all the experiments._
```bash  
# The main experiment implemented in your baseline using default hyperparameters (that should be setup in the Hydra configs) should run (including dataset download and necessary partitioning) by executing the command:

poetry run -m <baseline-name>.main <no additional arguments> # where <baseline-name> is the name of this directory and that of the only sub-directory in this directory (i.e. where all your source code is)

# If you are using a dataset that requires a complicated download (i.e. not using one natively supported by TF/PyTorch) + preprocessing logic, you might want to tell people to run one script first that will do all that. Please ensure the download + preprocessing can be configured to suit (at least!) a different download directory (and use as default the current directory). The expected command to run to do this is:

poetry run -m <baseline-name>.dataset_preparation <optional arguments, but default should always run>

# It is expected that you baseline supports more than one dataset and different FL settings (e.g. different number of clients, dataset partitioning methods, etc). Please provide a list of commands showing how these experiments are run. Include also a short explanation of what each one does. Here it is expected you'll be using the Hydra syntax to override the default config.

poetry run -m <baseline-name>.main  <override_some_hyperparameters>
.
.
.
poetry run -m <baseline-name>.main  <override_some_hyperparameters>
```


## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
