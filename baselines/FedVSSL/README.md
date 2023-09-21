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

![](docs/FVSSL.png)

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
To run FedVSSL with UCF-101 baseline, first follow the instruction in `dataset_preparation.py` to download the dataset, complete pre-processing and data partitioning for FL.
Then, ensure you have activated your Poetry environment (execute `poetry shell` from this directory), then:

```bash
# run federated SSL training with FedVSSL
python -m FedVSSL.main --pre_train=True # this will run using the default settings.

# you can override settings directly from the command line
python -m FedVSSL.main --mix_coeff=1 --rounds=100 # will set hyper-parameter alpha to 1 and the number of rounds to 100

# run downstream fine-tuning with pre-trained SSL model
python -m FedVSSL.main --pre_train=False # this will run using the default settings.

```

To run using FedAvg:
```bash
# this will run FedAvg baseline
# This is done so to match the experimental setup in the paper
python -m FedVSSL.main --fedavg=True

# this config can also be overriden.
```

## Expected results

### Pre-training results on UCF-101
The pre-training in the paper was conducted on Kinectics-400, which would take too much resource to complete the training.
As a result, we provide the following command with pre-training on UCF-101, in order to validate FedVSSL.

```bash
python -m FedVSSL.main --pre_train=True # this will run using the default settings.
```

To check the results, please direct to `ucf_FedVSSL_results/clientN/*.log.json` files in default, and check the loss changes during training.
If you have enough resource, feel free to try with Kinectics-400 following `data_partitioning_k400.py` for data partitioning.

### Downstream fine-tuning results on UCF-101

We provide the checkpoints of the pre-trained SSL models on Kinectics-400.
With them as starting points, we can run downstream fine-tuning on UCF-101 to obtain the expected results in the paper.

```bash
python -m FedVSSL.main --pre_train=False --pretrained_model_path=/path/to/checkpoints

# following the table below to change the checkpoints path.
```

| Method  | Checkpoint file                                                                                     | UCF R@1 |
|---------|-----------------------------------------------------------------------------------------------------|---------|
|FedVSSL$(\alpha=0, \beta=0)$ | [round-540.npz](https://drive.google.com/file/d/15EEIQay5FRBMloEzt1SQ8l8VjZFzpVNt/view?usp=sharing) | 34.34 |
|FedVSSL$(\alpha=1, \beta=0)$ | [round-540.npz](https://drive.google.com/file/d/1OUj8kb0ahJSKAZEB-ES94pOG5-fB-28-/view?usp=sharing) | 34.23 |
|FedVSSL$(\alpha=0, \beta=1)$ | [round-540.npz](https://drive.google.com/file/d/1N62kXPcLQ_tM45yd2kBYjNOskdHclwLM/view?usp=sharing) | 35.61 |
|FedVSSL$(\alpha=1, \beta=1)$ | [round-540.npz](https://drive.google.com/file/d/1SKb5aXjpVAeWbzTKMFN9rjHW_LQsmUXj/view?usp=sharing) | 35.66 |
|FedVSSL$(\alpha=0.9, \beta=0)$| [round-540.npz](https://drive.google.com/file/d/1W1oCnLXX0UJhQ4MlmRw-r7z5DTCeO75b/view?usp=sharing) |35.50|
|FedVSSL$(\alpha=0.9, \beta=1)$| [round-540.npz](https://drive.google.com/file/d/1BK-bbyunxTWNqs-QyOYiohaNv-t3-hYe/view?usp=sharing) |35.34|





