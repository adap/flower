Contribute Baselines
====================

Do you have a new federated learning paper and want to add a new baseline to Flower? Or do you want to add a new experiment to an already existing baseline paper? Great, we really appreciate your contribution.

The goal of Flower Baselines is to reproduce experiments from popular papers to accelerate researchers by enabling faster comparisons to new strategies, datasets, models, and federated pipelines in general. 

Before you start to work on a new baseline or experiment please check the `Flower Issues <https://github.com/adap/flower/issues>`_ or `Flower Pull Requests <https://github.com/adap/flower/pulls>`_ to see if someone else is already working on it. Please open a new issue if you are planning to work on a new baseline or experiment with a short description of the corresponding paper and the experiment you want to contribute.


Requirements
------------

Contributing a new baseline is really easy. You only have to make sure that your federated learning experiments are running with Flower. As soon as you have created a Flower-based experiment, you can contribute it.

It is recommended (but not required) to use `Hydra <https://hydra.cc/>`_ to execute the experiment. 

Please make sure that you add your baseline or experiment to the corresponding directory as explained in `Execute Baseline <https://flower.dev/docs/execute-baseline.html>`_. Also add a :code:`README.md` with a short explanation of the paper and a detailed description of how to execute the baseline. 
Please also check if all requirements or library are listed in the :code:`pyproject.toml` or the :code:`requirements.txt` available in the directory `flower/baselines/flwr_baselines <https://github.com/adap/flower/blob/main/baselines>`_. If the requirement or library is not yet listed please add it. If the versions are different please make your that your baseline is running with the already existing version listed in :code:`pyproject.toml` or the :code:`requirements.txt`.

The experiment also needs to contain a file with an automatic downloader of the dataset. This can be included in one of the files or as an extra file.

At the end please also add a plot the results of the experiment that are performed by your code to the experiment directory as well as to the :code:`README.md`. This will help to better understand the experiment and allow the user to figure out any errors.  

We are aware that a few libraries are only available within the Conda environment. However, please make sure that your code runs as well outside of Conda to keep the usability of the Flower framework. 

Here is a checklist for adding a new baseline:

* add required Python packages to :code:`pyproject.toml` or :code:`requirements.txt`
* add all required code under :code:`baselines/flwr_baselines/publications/[new_publication]`
* add a dataset downloader
* add an experiment plot
* add a :code:`README.md`

Usability
---------

Flower is known and loved for its usability. Therefore, make sure that your baseline or experiment can be executed with a single command such as :code:`./run.sh` or :code:`python3 main.py`. How you organise the experiments and the related code structure is up to you as an author, but please keep in mind to make sure that other users can easily understand and execute your baseline.

We look forward to your contribution!
