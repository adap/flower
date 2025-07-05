---
title: FedBABU: Towards Enhanced Representation for Federated Image Classification
url: https://arxiv.org/abs/2106.06042
labels: [image classification, label heterogeneity, personalized federated learning]
dataset: [CIFAR-10]
---

# FedBABU: Towards Enhanced Representation for Federated Image Classification

**Paper:** [arxiv.org/abs/2106.06042](https://arxiv.org/abs/2106.06042)

**Authors:** Jaehoon Oh, Sangmook Kim, Se-Young Yun

**Reproduction & Implementation:** Jiahao Tan<<karhoutam@qq.com>>

**Abstract:**
Federated learning has evolved to improve a single global model under data heterogeneity (as a curse) or to develop multiple personalized models using data heterogeneity (as a blessing). However, little research has considered both directions simultaneously. This paper investigates the relationship between global model performance and personalization, showing that better global models do not always yield better personalization. The authors decompose the network into a body (feature extractor) and a head (classifier), and identify that training the head can degrade personalization. Based on this, they propose FedBABU, a federated learning algorithm that only updates the body during federated training, while the head is randomly initialized and only fine-tuned locally for personalization during evaluation. Experiments demonstrate consistent improvements and efficient personalization with FedBABU.


## About this baseline

**What’s implemented:**
This codebase replicates the experiments in _FedBABU: Towards Enhanced Representation for Federated Image Classification_ (Jaehoon Oh et al., 2021) for the CIFAR-10 dataset, specifically the setting with 100 clients and 2 classes per client (see Table 13 in the paper). The implementation follows the FedBABU algorithm, which separates the model into a body (shared feature extractor) and a head (personalized classifier), updating only the body during federated training and fine-tuning the head locally for each client during evaluation.

**Datasets:** CIFAR-10 from `Flower Datasets`.

**Contributors:** Jiahao Tan<<karhoutam@qq.com>>


## Experimental Setup

**Task:** Image Classification

**Model:** This directory implements the FourConvNet architecture as described in Appendix A.1 of the paper:
- FourConvNet

**Dataset:** CIFAR-10, partitioned such that each client receives data from 2 unique classes (non-IID split).

**FedBABU Algorithm:**
- During federated training, only the body (feature extractor) of the model is updated; the head (classifier) is randomly initialized and not updated.
- During evaluation, the head is fine-tuned locally on each client to personalize the model.
- This approach aims to learn universal representations in the body while enabling efficient personalization via the head.

**Training Hyperparameters:** The hyperparameters can be found in the `pyproject.toml` file under the `[tool.flwr.app.config]` section.

| Description           | Default Value                       |
| --------------------- | ----------------------------------- |
| `num-server-rounds`   | `80`                                |
| `num-local-epochs`    | `4`                                 |
| `num-finetune-epochs` | `5`                                 |
| `client-resources`    | `{'num-cpus': 2, 'num-gpus': 0.5 }` |
| `learning-rate`       | `0.1`                               |
| `batch-size`          | `50`                                |
| `model-name`          | `FourConvNet`                       |
| `algorithm`           | `fedbabu`                           |


## Environment Setup

Create a new Python environment and install the baseline project:

```bash
pip install -e .
```

## Running the Experiments

```
flwr run . # this will run using the default settings in the `pyproject.toml`
```

While the config files contain a large number of settings, the ones below are the main ones you'd likely want to modify:
```bash
algorithm = "fedavg", "fedbabu" # these are currently supported
num_classes_per_client = 2
```

## Expected Results
The default algorithm used by all configuration files is `fedbabu`. To use `fedavg`, change the `algorithm` property in the respective configuration file. The default federated environment consists of 100 clients. The paper's results of CIFAR-10 are listed at Appendix F.2.

### CIFAR-10 (100, 2)

```
flwr run . 
```

**Result:** Check [`plot.ipynb`](./plot.ipynb)