# Vertical Federated Learning example

This example will showcase how you can perform Vertical Federated Learning using
Flower. We'll be using the [Titanic dataset](https://www.kaggle.com/competitions/titanic/data) 
to train simple regression models for binary classification. We will go into
more details below, but the main idea of Vertical Federated Learning is that
each clients are holding different feature sets of the same <TODO> and that the
server is holding the labels of this <TODO>.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you
can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/vertical-fl . && rm -rf _tmp && cd vertical-fl
```

This will create a new directory called `vertical-fl` containing the
following files:

```shell
-- pyproject.toml
-- requirements.txt
-- docs/data/train.csv
-- client.py
-- plot.py
-- simulation.py
-- strategy.py
-- task.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in
`pyproject.toml` and `requirements.txt`. We recommend
[Poetry](https://python-poetry.org/docs/) to install those dependencies and
manage your virtual environment ([Poetry
installation](https://python-poetry.org/docs/#installation)) or
[pip](https://pip.pypa.io/en/latest/development/), but feel free to use a
different way of installing dependencies and managing virtual environments if
you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual
environment. To verify that everything works correctly you can run the following
command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according
to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Usage

Once everything is installed, you can just run:

```shell
poetry run python3 simulation.py
```

for `poetry`, otherwise just run:

```shell
python3 simulation.py
```

This will start the Vertical FL training for 500 rounds with 3 clients.
Eventhough the number of rounds is quite high, this should only take a few
seconds to run as the model is very small.

## Explanations

### Vertical FL vs Horizontal FL

|                       | Horizontal Federated Learning (HFL)                                                                                                                                                                      | Vertical Federated Learning (VFL)                                                                                                                                                                                                                                                                                                                                        |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data Distribution     | Clients have different data instances but share the same feature space.  Think of different hospitals having different patients' data (samples)  but recording the same types of information (features). | Each client holds different features for the same instances.  Imagine different institutions holding various tests or  measurements for the same group of patients.                                                                                                                                                                                                      |
| Model Training        | Each client trains a model on their local data,  which contains all the feature columns for its samples.                                                                                                 | Clients train models on their respective features without  having access to the complete feature set.  Each model only sees a vertical slice of the data (hence the name 'Vertical').                                                                                                                                                                                    |
| Aggregation           | The server aggregates these local models by averaging  the parameters or gradients to update a global model.                                                                                             | The server aggregates the updates such as gradients or parameters,  which are then used to update the global model.  However, since each client sees only a part of the features,  the server typically has a more complex role,  sometimes needing to coordinate more sophisticated aggregation strategies  that may involve secure multi-party computation techniques. |
| Privacy Consideration | The raw data stays on the client's side, only model updates are shared,  which helps in maintaining privacy.                                                                                             | VFL is designed to ensure that no participant can access  the complete feature set of any sample,  thereby preserving the privacy of data.                                                                                                                                                                                                                               |

### Data

#### Titanic dataset

Context, Features, preprocessing, target, nb samples

#### Partitioning

What do we give to each client?

### Models

What model does the server and clients train on?

### Strategy

What strategy are we using for the aggregation?
