---
title: Federated Learning for ASR based on Wav2vec2.0
url: https://ieeexplore.ieee.org/document/10096426
labels: [speech, asr]
dataset: [TED-LIUM 3]
---

# Federated Learning for ASR Based on wav2vec 2.0

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [ieeexplore.ieee.org/document/10096426](https://ieeexplore.ieee.org/document/10096426)

**Authors:** Tuan Nguyen, Salima Mdhaffar, Natalia Tomashenko, Jean-François Bonastre, Yannick Estève

**Abstract:** This paper presents a study on the use of federated learning to train an ASR model based on a wav2vec 2.0 model pre-trained by self supervision. Carried out on the well-known TED-LIUM 3 dataset, our experiments show that such a model can obtain, with no use of a language model, a word error rate of 10.92% on the official TEDLIUM 3 test set, without sharing any data from the different users. We also analyse the ASR performance for speakers depending to their participation to the federated learning. Since federated learning was first introduced for privacy purposes, we also measure its ability to protect speaker identity. To do that, we exploit an approach to analyze information contained in exchanged models based on a neural network footprint on an indicator dataset. This analysis is made layer-wise and shows which layers in an exchanged wav2vec 2.0 based model bring the speaker identity information.


## About this baseline

**What’s implemented:** Figure 1 in the paper.

**Datasets:** TED-LIUM 3 dataset. It's requires a 54GB download. Once extracted it is ~60 GB. You can read more about this dataset in the [TED-LIUM 3](https://arxiv.org/abs/1805.04699) paper. A more concise description of this dataset can be found in the [OpenSLR](https://www.openslr.org/51/) site.

**Hardware Setup:** Training `wav2vec2.0` is a bit memory intensive so you'd need at least a 24GB GPU. With the current settings, each client requires ~15GB of VRAM. This suggest you could run the experiment fine on a 16GB GPU but not if you also need to pack the global model evaluation stage on the same GPU. On a single RTX 3090Ti (24GB VRAM) each round takes between 20 and 40 minutes (depending on which clients are sampled, some clients have more data than others).

**Contributors:** Tuan Nguyen


## Experimental Setup

**Task:** Automatic Speech Recognition (ASR)

**Model:** Wav2vec2.0-large [from Huggingface](https://huggingface.co/facebook/wav2vec2-large-lv60) totalling 317M parameters. Read more in the [wav2vec2.0 paper](https://arxiv.org/abs/2006.11477).

**Dataset:** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

**Training Hyperparameters:** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

Once you have installed `pyenv` and `poetry`, run the commands below to setup your python environment:

```bash
# Set a recent version of Python for your environment
pyenv local 3.10.6
poetry env use 3.10.6

# Install your environment
poetry install

# Activate your environment
poetry sehll
```

```bash
# Then create a directory using the same name as you'll use for `dada_dir` in your config (see conf/base.yaml)
mkdir data

# Clone client mapping (note content will be moved to your data dir)
git clone https://github.com/tuanct1997/Federated-Learning-ASR-based-on-wav2vec-2.0.git _temp && mv _temp/data/* data/ && rm -rf _temp

# Download dataset, extract and prepare dataset partitions
python -m fedwav2vec2.dataset_preparation
```


## Running the Experiments

```bash
python -m fedwav2vec2.main # will run on Flower client per GPU

# if you have a large GPU (32GB+) you migth want to fit two per GPU
python -m fedwav2vec2.main client_resources.num_gpus=0.5
```


## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run python -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
