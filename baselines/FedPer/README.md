---
title: Federated Learning with Personalization Layers
url: https://arxiv.org/abs/1912.00818
labels: ["system heterogeneity", "image classification", "personalization", "horizontal data partition"] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: ["CIFAR-10", "FLICKR-AES"] # list of datasets you include in your baseline
---

# Federated Learning with Personalization Layers

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

****Paper:**** : https://arxiv.org/abs/1912.00818

****Authors:**** : Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and Sunav Choudhary

****Abstract:**** : The emerging paradigm of federated learning strives to enable collaborative training of machine learning models on the network edge without centrally aggregating raw data and hence, improving data privacy. This sharply deviates from traditional machine learning and necessitates design of algorithms robust to various sources of heterogeneity. Specifically, statistical heterogeneity of data across user devices can severely degrade performance of standard federated averaging for traditional machine learning applications like personalization with deep learning. This paper proposes `FedPer`, a base + personalization layer approach for federated training of deep feed forward neural networks, which can combat the ill-effects of statistical heterogeneity. We demonstrate effectiveness of `FedPer` for non-identical data partitions of CIFAR datasets and on a personalized image aesthetics dataset from Flickr.

## About this baseline

****What’s implemented:**** : The code in this directory replicates the experiments in _Federated Learning with Personalization Layers_ (Arivazhagan et al., 2019) for CIFAR10 and FLICKR-AES datasets, which proposed the `FedPer` model. Specifically, it replicates the results found in figures 2, 4, 7, and 8 in their paper. __Note__ that there is typo in the caption of Figure 4 in the article, it should be CIFAR10 and __not__ CIFAR100. 

****Datasets:**** : CIFAR10 from PyTorch's Torchvision and FLICKR-AES. FLICKR-AES was proposed as dataset in _Personalized Image Aesthetics_ (Ren et al., 2017) and can be downloaded using a link provided on thier [GitHub](https://github.com/alanspike/personalizedImageAesthetics). One must first download FLICKR-AES-001.zip (5.76GB), extract all inside and place in baseline/FedPer/datasets. To this location, also download the other 2 related files: (1) FLICKR-AES_image_labeled_by_each_worker.csv, and (2) FLICKR-AES_image_score.txt. Current repository supports CIFAR100 but it hasn't been used to reproduce figures. 

****Hardware Setup:**** : Experiments have been carried out on GPU. 2 different computers managed to run experiments: 

- GeForce RTX 3080 16GB
- GeForce RTX 4090 24GB

****Contributors:**** : William Lindskog


## Experimental Setup

****Task:**** : Image Classification

****Model:**** : This directory implements 2 models:

- ResNet34 which can be imported directly (after having installed the packages) from PyTorch, using `from torchvision.models import resnet34 
- MobileNet-v1

Please see how models are implemented using a so called model_manager and model_split class since FedPer uses head and base layers in a neural network. These classes are defined in the models.py file and thereafter called when building new models in the directory /implemented_models. Please, extend and add new models as you wish. 

****Dataset:**** : CIFAR10, FLICKR-AES. CIFAR10 will be partitioned based on number of classes for data that each client shall recieve e.g. 4 allocated classes could be [1, 3, 5, 9]. FLICKR-AES is an unbalanced dataset, so there we only apply random sampling. 

****Training Hyperparameters:**** : The hyperparameters can be found in conf/base.yaml file which is the configuration file for the main script. 

| Description | Default Value |
| ----------- | ----- |
| num_clients | 10 |
| clients per round | 10 |
| number of rounds | 50 |
| client resources | {'num_cpus': 32, 'num_gpus': 1 }|
| learning_rate | 0.01 |
| batch_size | 128 |
| optimizer | SGD |
| algorithm | fedavg|


## Environment Setup

To construct the Python environment follow these steps:

```bash
# install the base Poetry environment
poetry install

# activate the environment
poetry shell

# install PyTorch with GPU support. Please note this baseline is very lightweight so it can run fine on a CPU.
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 
```
## Running the Experiments
```bash
python -m FedPer.main # this will run using the default settings in the `conf/base.yaml`    
```

To reproduce figures:
```bash
# make FedPer/run_scripts.sh executable
chmod u+x FedPer/run_scripts.sh
# uncomment lines in script that you want to run, then  
bash FedPer/run_scripts.sh

# this config can also be overriden from the CLI
```

What you need to change in configuration files: 
```bash
algorithm: fedavg, fedper # these are currently supported
server_device: 'cuda:0', 'cpu'
dataset.name: 'cifar10', 'cifar100', 'flickr'
num_classes: 10, 100, 5 # respectively 
model.name: mobile, resnet
```


## Expected Results

Having run the `run_script.sh`, the expected results should look something like this: 

__MobileNet-v1 on CIFAR10__

![](FedPer/visuals/use/mobile_plot_figure_2.png)

__ResNet on CIFAR10__

![](FedPer/visuals/use/resnet_plot_figure_2.png)

__MobileNet-v1 on CIFAR10 using varying size of head__

![](FedPer/visuals/use/mobile_plot_figure_num_head.png)


__ResNet on CIFAR10 using varying size of head__

![](FedPer/visuals/use/resnet_plot_figure_num_head.png)

__MobileNet-v1 on FLICKR-AES__

![](FedPer/visuals/use/mobile_plot_figure_flickr.png)

__ResNet on FLICKR-AES__

![](FedPer/visuals/use/resnet_plot_figure_flickr.png)