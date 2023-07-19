Contributing Baselines
======================

Do you have a new federated learning paper and want to add a new baseline to Flower? Or do you want to add an experiment to an existing baseline paper? Great, we really appreciate your contribution.

The goal of Flower Baselines is to reproduce experiments from popular papers to accelerate researchers by enabling faster comparisons to new strategies, datasets, models, and federated pipelines in general. 

Before you start to work on a new baseline or experiment, please check the `Flower Issues <https://github.com/adap/flower/issues>`_ or `Flower Pull Requests <https://github.com/adap/flower/pulls>`_ to see if someone else is already working on it. Please open a new issue if you are planning to work on a new baseline or experiment with a short description of the corresponding paper and the experiment you want to contribute.

TL;DR: Adding a new Flower Baseline
-----------------------------------
.. warning::
    We are in the process of changing how Flower Baselines are structured and updating the instructions for new contributors. Bear with us until we have finalised this transition. For now, follow the steps described below and reach out to us if something is not clear. We look forward to welcoming your baseline into Flower!!
.. note::
    For a detailed set of steps to follow, check the `Baselines README on GitHub <https://github.com/adap/flower/tree/main/baselines>`_.

Let's say you want to contribute the code of your most recent Federated Learning publication, *FedAwesome*. There are only three steps necessary to create a new *FedAwesome* Flower Baseline:

#. **Get the Flower source code on your machine**
    #. Fork the Flower codebase: go to the `Flower GitHub repo <https://github.com/adap/flower>`_ and fork the code (click the *Fork* button in the top-right corner and follow the instructions)
    #. Clone the (forked) Flower source code: :code:`git clone git@github.com:[your_github_username]/flower.git`
    #. Open the code in your favorite editor.
#. **Create a directory for your baseline and add the FedAwesome code**
    #. Navigate to the baselines directory and run :code:`./dev/create-baseline.sh fedawesome`
    #. A new directory in :code:`baselines/fedawesome` is created.
    #. Follow the instructions in :code:`EXTENDED_README.md` and :code:`README.md` in :code:`baselines/fedawesome/`. 
#. **Open a pull request**
    #. Stage your changes: :code:`git add .`
    #. Commit & push: :code:`git commit -m "Create new FedAwesome baseline" ; git push`
    #. Open a pull request: go to *your* fork of the Flower codebase and create a pull request that targets the Flower ``main`` branch
    #. Interact with the Flower maintainers during the merging process and make adjustments to your code as needed.

Further reading:

* `GitHub docs: About forks <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks>`_
* `GitHub docs: Creating a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_
* `GitHub docs: Creating a pull request from a fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_

Requirements
------------

Contributing a new baseline is really easy. You only have to make sure that your federated learning experiments are running with Flower and replicate the results of a paper. 

The only requirement you need in your system in order to create a baseline is to have `Poetry <https://python-poetry.org/docs/>`_ installed. This is our package manager tool of choice. 

We are adopting `Hydra <https://hydra.cc/>`_ as the default mechanism to manage everything related to config files and the parameterisation of the Flower baseline.

Usability
---------

Flower is known and loved for its usability. Therefore, make sure that your baseline or experiment can be executed with a single command such as :code:`conda run -m <your-baseline>.main` or :code:`python main.py` (when sourced into your environment). We provide you with a `template-baseline <https://github.com/adap/flower/tree/main/baselines/baseline_template>`_ to use as guidance when contributing your baseline. Having all baselines follow a homogenous structure helps users to tryout many baselines without the overheads of having to understand each individual codebase. Similarly, by using Hydra throughout, users will immediately know how to parameterise your experiments directly from the command line.

We look forward to your contribution!
