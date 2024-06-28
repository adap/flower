---
title: Flower Example using MONAI
labels: [quickstart, medical, vision]
dataset: [MedNIST]
framework: [`MONAI <https://monai.io/>`_]
---

# Flower Example using MONAI

This introductory example to Flower uses MONAI, but deep knowledge of MONAI is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy.

[MONAI](https://docs.monai.io/en/latest/index.html)(Medical Open Network for AI) is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of the PyTorch Ecosystem.

Its ambitions are:

- developing a community of academic, industrial and clinical researchers collaborating on a common foundation;

- creating state-of-the-art, end-to-end training workflows for healthcare imaging;

- providing researchers with an optimized and standardized way to create and evaluate deep learning models.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/quickstart-monai . && rm -rf _tmp && cd quickstart-monai
```

This will create a new directory called `quickstart-monai` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- data.py
-- model.py
-- server.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `monai` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run Federated Learning with MONAI and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminal windows and run the following commands.  Clients will train a [DenseNet121](https://docs.monai.io/en/stable/networks.html#densenet121) from MONAI. If a GPU is present in your system, clients will use it.

Start client 1 in the first terminal:

```shell
python3 client.py --partition-id 0
```

Start client 2 in the second terminal:

```shell
python3 client.py --partition-id 1
```

You will see that the federated training is starting. Look at the [code](https://github.com/adap/flower/tree/main/examples/quickstart-monai) for a detailed explanation.
