---
title: title of the paper
url: https://arxiv.org/abs/2007.14390 # update with the link to your paper
labels: [label1, label2] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. system heterogeneity, image classification, asynchronous, weight sharing, cross-silo). Do not use "". Remove this comment once you are done.
dataset: [dataset1, dataset2] # list of datasets you include in your baseline. Do not use "". Remove this comment once you are done.
---

> [!IMPORTANT]
> This is the template for your `README.md`. Please fill-in the information in all areas witha :warning: symbol.
> Please refer to the [Flower Baselines contribution](https://flower.ai/docs/baselines/how-to-contribute-baselines.html) and [Flower Baselines usage](https://flower.ai/docs/baselines/how-to-use-baselines.html) guides for more details.
> Please complete the metadata section at the very top of this README. This generates a table at the top of the file that will facilitate indexing baselines.
> You can check the [MOON baseline](https://github.com/adap/flower/tree/main/baselines/moon) as an example of a baseline that followed this guide.
> Please remove this [!IMPORTANT] block once you are done with your `README.md` as well as all the `:warning: symbols

# :warning: *_Title of your baseline_*

> [!NOTE] 
> If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** :warning: *_add the URL of the paper page (not to the .pdf). For instance if you link a paper on ArXiv, add here the URL to the abstract page (e.g. [paper](https://arxiv.org/abs/1512.03385)). If your paper is in from a journal or conference proceedings, please follow the same logic._*

**Authors:** :warning: *_list authors of the paper_*

**Abstract:** :warning: *_add here the abstract of the paper you are implementing_*


## About this baseline

**What’s implemented:** :warning: *_Concisely describe what experiment(s) (e.g. Figure 1, Table 2, etc) in the publication can be replicated by running the code. Please only use a few sentences. ”_*

**Datasets:** :warning: *_List the datasets you used (if you used a medium to large dataset, >10GB please also include the sizes of the dataset). We highly recommend using [FlowerDatasets](https://flower.ai/docs/datasets/index.html) to download and partition your dataset. If you have other ways to download the data, you can also use `FlowerDatasets` to partiion it._*

**Hardware Setup:** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Indicate how long it took to run the experiments. Someone out there might not have access to the same resources you have so, could you list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

**Contributors:** :warning: *_let the world know who contributed to this baseline. This could be either your name, your name and affiliation at the time, or your GitHub profile name if you prefer. If multiple contributors signed up for this baseline, please list yourself and your colleagues_*


## Experimental Setup

**Task:** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

**Model:** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

**Dataset:** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

**Training Hyperparameters:** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

:warning: _The Python environment for all baselines should follow these guidelines in the `EXTENDED_README`. Specify the steps to create and activate your environment. If there are any external system-wide requirements, please include instructions for them too. These instructions should be comprehensive enough so anyone can run them (if non standard, describe them step-by-step)._


## Running the Experiments

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to override some arguments.
flwr run . --run-config learning-rate=0.1,coefficient=0.123

# or you might want to load different `.toml` configs all together:
flwr run . --run-config <my-big-experiment-config>.toml
```

:warning: _It is preferable to show a single commmand (or multilple commands if they belong to the same experiment) and then a table/plot with the expected results, instead of showing all the commands first and then all the results/plots._
:warning: _If you present plots or other figures, please include either a Jupyter notebook showing how to create them or include a utility function that can be called after the experiments finish running._
:warning: If you include plots or figures, save them in `.png` format and place them in a new directory named `_static` at the same level as your `README.md`.
