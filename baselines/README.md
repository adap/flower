# Flower Baselines


> We are changing the way we structure the Flower baselines. While we complete the transition to the new format, you can still find the existing baselines in the `flwr_baselines` directory. Currently, you can make use of baselines for [FedAvg](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/fedavg_mnist), [FedOpt](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/adaptive_federated_optimization), and [LEAF-FEMNIST](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist).

> The documentation below has been updated to reflect the new way of using Flower baselines.


## Structure

Each baseline in this directory is fully self-contained in terms of source code in its own directory. In addition, each baseline uses its very own Python environment as designed by the contributors of such baseline in order to replicate the experiments in the paper. Each baseline directory contains the following structure:

```bash
baselines/<baseline-name>/
                ├── README.md
                ├── pyproject.toml
                └── <baseline-name>
                            ├── *.py # several .py files including main.py and __init__.py
                            └── conf
                                └── *.yaml # one or more Hydra config files
```
Please note that some baselines might include additional files (e.g. a `requirements.txt`) or a hierarchy of `.yaml` files for [Hydra](https://hydra.cc/).

## Running the baselines

Each baseline is self-contained in its own directory. Furthermore, each baseline defines its own Python environment using [Poetry](https://python-poetry.org/docs/) via a `pyproject.toml` file and  [`pyenv`](https://github.com/pyenv/pyenv). If you haven't setup `Poetry` and `pyenv` already on your machine, please take a look at the [Documentation](https://flower.ai/docs/baselines/how-to-use-baselines.html#setting-up-your-machine) for a guide on how to do so.

Assuming `pyenv` and `Poetry` are already installed on your system. Running a baseline can be done by:

1. Cloning the flower repository

    ```bash
    git clone https://github.com/adap/flower.git && cd flower
    ```

2. Navigate inside the directory of the baseline you'd like to run.
3. Follow the `[Environment Setup]` instructions in the `README.md`. In most cases this will require you to just do:

    ```bash
    poetry install
    ```
4. Run the baseline as indicated in the `[Running the Experiments]` section in the `README.md` or in the `[Expected Results]` section to reproduce the experiments in the paper.


## Contributing a new baseline

Do you have a new federated learning paper and want to add a new baseline to Flower? Or do you want to add an experiment to an existing baseline paper? Great, we really appreciate your contribution !!

The steps to follow are:

1. Fork the Flower repo and clone it into your machine.
2. Navigate to the `baselines/` directory, choose a single-word (and **lowercase**) name for your baseline, and from there run:

    ```bash
    # This will create a new directory with the same structure as `baseline_template`.
    ./dev/create-baseline.sh <baseline-name>
    ```
3. Then, go inside your baseline directory and continue with the steps detailed in `EXTENDED_README.md` and `README.md`.
4. Once your code is ready and you have checked that following the instructions in your `README.md` the Python environment can be created correctly and that running the code following your instructions can reproduce the experiments in the paper, you just need to create a Pull Request (PR). Then, the process to merge your baseline into the Flower repo will begin!


Further resources:
* [GitHub docs: About forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)
* [GitHub docs: Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
* [GitHub docs: Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)

