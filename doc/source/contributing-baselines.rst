Contributing Baselines
======================

Do you have a new federated learning paper and want to add a new baseline to Flower? Or do you want to add a new experiment to an already existing baseline paper? Great, we really appreciate your contribution.

The goal of Flower Baselines is to reproduce experiments from popular papers to accelerate researchers by enabling faster comparisons to new strategies, datasets, models, and federated pipelines in general. 

Before you start to work on a new baseline or experiment please check the `Flower Issues <https://github.com/adap/flower/issues>`_ or `Flower Pull Requests <https://github.com/adap/flower/pulls>`_ to see if someone else is already working on it. Please open a new issue if you are planning to work on a new baseline or experiment with a short description of the corresponding paper and the experiment you want to contribute.

TL;DR: Adding a new Flower Baseline
-----------------------------------

Let's say you want to contribute the code of your most recent Federated Learning publication, *FedAweseome*. There are only three steps necessary to create a new *FedAweseome* Flower Baseline:

#. **Get the Flower source code on your machine**
    #. Fork the Flower codebase: got to the `Flower GitHub repo <https://github.com/adap/flower>`_ and fork the code (click the *Fork* button in the top-right corner and follow the instructions)
    #. Clone the (forked) Flower source code: :code:`git clone git@github.com:[your_github_username]/flower.git`
    #. Open the code in your favorite editor (e.g., using VSCode: ``cd flower ; code .``)
#. **Add the FedAwesome code**
    #. Add your :code:`FedAwesome` code under :code:`baselines/flwr_baselines/publications/[fedawesome]`
    #. Add a `pyproject.toml` with all necessary dependencies
    #. Add a `README.md` describing how to use your baseline
#. **Open a pull request**
    #. Stage your changes: :code:`git add .`
    #. Commit & push: :code:`git commit -m "Create new FedAweseome baseline" ; git push`
    #. Open a pull request: go to *your* fork of the Flower codebase and create a pull request that targets the Flower ``main``` branch

Further reading:

* `GitHub docs: About forks <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks>`_
* `GitHub docs: Creating a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_
* `GitHub docs: Creating a pull request from a fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_

Requirements
------------

Contributing a new baseline is really easy. You only have to make sure that your federated learning experiments are running with Flower. As soon as you have created a Flower-based experiment, you can contribute it.

It is recommended (but not required) to use `Hydra <https://hydra.cc/>`_ to execute the experiment. 

Please make sure to add your baseline or experiment to the corresponding directory as explained in `Executing Baseline <https://flower.dev/docs/using-baselines.html>`_. Give your baseline the unique identifier, for example, :code:`fedbn` refers to the paper "FedBN: Federated Learning on non-IID Features via Local Batch Normalization" and create the corresponding directory :code:`flower/baselines/flwr_baselines/publications/fedbn`. Then you create the experiment directory with the experiment name, for example the epxeriment that measures the convergence has the directory :code:`flower/baselines/flwr_baselines/publications/fedbn/convergence_rate`. This directory contains all your code and a :code:`README.md` with a link to the paper, the abstract of the paper, and a detailed description of how to execute the experiments.

Please also check if all required Python packages (libraries, frameworks, ...) are listed in :code:`pyproject.toml` and/or :code:`requirements.txt` (all in the directory `baselines <https://github.com/adap/flower/blob/main/baselines>`_. If the required Python package is not yet listed, please add it to :code:`pyproject.toml`. If you need a different version of a package that is already listed, please try to make your experiment running with the already existing version listed in :code:`pyproject.toml` (or :code:`requirements.txt`) or - if that does not work - open a GitHub Issue and request the version change.

The experiment also needs to contain a file with a (if possible automatic) downloader of the dataset. This can be included in one of the files or as an extra file.

Finally, please also add plots of the results of the experiments that are performed by your code to the experiment directory and include them in :code:`README.md`. This will help others to better understand the experiment and allow them to quickly recognise your contributions.

We are aware that a small number of libraries are only available via Conda. However, we want to encourage you to make sure that your code also runs well outside of Conda in order to make it more accessible to the wider research community.

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
