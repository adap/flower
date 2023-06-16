# **:warning:***_Title of your baseline_*

> :warning: Please follow the instructions carefully. First copy this directory and rename it with something meaningful about your baseline. Then add the code into the Python scripts provided, and edit all the fields in the readme that start with a :warning: symbols (and remove the comments as you start adding the information). You can see the [FedProx-Mnist baseline](~https://github.com/adap/flower/tree/fedprox_mnist_refresh/baselines/flwr_baselines/publications/fedprox_mnist~) as an example of a baseline that followed this guide.


****Labels:**** :warning: *_add a list of single-word (maybe two-words) terms that can be used to categorise this baseline. For example: `image classification`, `heterogeneous clients`, `personalisation`, `communication efficiency`. Please use between 4 and 10 labels._*

****Paper:**** :warning: *_add the URL of the paper._*

****Authors:**** :warning: *_list authors of the paper_*

****Abstract:**** :warning: *_add here the abstract of the paper you are implementing_*

--------

## **About this baseline**

****What’s implemented:**** :warning: *_Concisely describe what experiment(s) in the publication can be replicated by running the code. Please only use a few sentences. Start with: “The code in this directory …”_*

****Datasets:**** :warning: *_List the datasets you used (if you used a medium to large dataset, >10GB please also include the sizes of the dataset)._*

****Hardware Setup:**** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

****Contributors:**** :warning: *_let the world know who contributed to this baseline. This could be either your name, your name and affiliation at the time, or your GitHub profile name if you prefer. If multiple contributors signed up for this baseline, please list yourself and your colleagues_*

-------
## **Experimental Setup**

****Task:**** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

****Model:**** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

****Dataset:**** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

****Training Hyperparameters:**** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*

-----
## **Running the Experiment**

:warning: _Provide instructions on the steps to follow to run all the experiments.
```bash  
# The main experiment implemented in your baseline using default hyperparameters (that should be setup in the Hydra configs) should run (including dataset download and necessary partitioning) by executing the command:

python main.py <no additional arguments>

# If you are using a dataset that requires a complicated download (i.e. not using one natively supported by TF/PyTorch) + preprocessing logic, you might want to tell people to run one script first that will do all that. Please ensure the download + preprocessing can be configured to suit (at least!) a different download directory (and use as default the current directory). The expected command to run to do this is:

python dataset_preparation.py <optional arguments, but default should always run>

# It is expected that you baseline supports more than one dataset and different FL settings (e.g. different number of clients, dataset partitioning methods, etc). Please provide a list of commands showing how these experiments are run. Include also a short explanation of what each one does. Here it is expected you'll be using the Hydra syntax to override the default config.

* python main.py <override_some_hyperparameters>
.
.
.
* python main.py <override_some_hyperparameters>
```

-----
## **Expected Results**

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Pleas add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

python main.py --multirun num_client_per_roun=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# ad more commands + plots for additional experiments.
```
