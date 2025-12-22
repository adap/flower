---
tags: [basic, tabular, federated analytics]
dataset: [artificial]
framework: [pandas]
---

# Federated Analytics with OMOP CDM using Flower

This example will show you how you can use Flower to run federated analytics workloads on distributed SQL databases, which are applicable to many biostatistics and healthcare use cases. You will use an artificial dataset generated in adherance to the [OMOP Common Data Model](https://www.ohdsi.org/data-standardization/), which uses the OHDSI standardized vocabularies widely adopted in clinical domains. You will also run this example with Flower's Deployment Engine to demonstrate how each SuperNode can be configured to connect to different PostgreSQL databases, respectively, while the database connection will be handled using SQLAlchemy and the `pyscopg` adapter which is the latest implementation of the PostgreSQL adapter for Python.

## Set up the project

### Clone the project

After cloning the project, this will create a new directory called `federated-analytics` containing the following files:

```shell
federated-analytics
├── db_init.sh          # Defines an artificial OMOP CDM table
├── db_start.sh         # Generates and starts PostgreSQL containers with OMOP CDM data
├── federated-analytics
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your database connection and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `federated-analytics` package.

```shell
# From a new python environment, run:
pip install -e .
```

### Start PostgreSQL databases

Run the following to start two PostgreSQL databases and initialize the dataset for each:

```shell
./db_start.sh
```

> [!NOTE]
> To start more than two databases, pass the desired number as the first argument to the script, e.g. `./db_start.sh 3`.

### Run with the Deployment Engine

For a basic execution of this federated analytics app, activate your environment and start the SuperLink process in insecure mode:

```shell
flower-superlink --insecure
```

Next, start two Supernodes and connect them to the SuperLink. You will need to specify different `--node-config` values so that each SuperNode will connect to different PostgreSQL databases.

```shell
flower-supernode \
     --insecure \
     --superlink 127.0.0.1:9092 \
     --clientappio-api-address 127.0.0.1:9094 \
     --node-config="db-port=5433"
```

```shell
flower-supernode \
     --insecure \
     --superlink 127.0.0.1:9092 \
     --clientappio-api-address 127.0.0.1:9095 \
     --node-config="db-port=5434"
```

Next, update the [`pyproject.toml`](./pyproject.toml) file to add a new federation configuration:

```toml
[tool.flwr.federations.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

Finally, run the Flower App and follow the `ServerApp` logs to track the execution of the run:

```shell
flwr run . federated-analytics --stream
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```shell
flwr run . federated-analytics --run-config "selected-features='age,bmi'" --stream
```

The steps above are adapted from this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html). After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
