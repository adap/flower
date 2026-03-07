# forest-monitoring-example: A Flower / PyTorch app

This Flower example uses PyTorch for a regression task, modelling forest timber volme using satellite time series data. The example is easy to run and helps illustrate how Flower can be adapted to your own use case. It uses a dummy demo datasets for example runs using the simulation engine and deployment mode.

## Fetch the App
Install Flower:

```bash
pip install flwr
```


This will create a new directory called forest-monitoring-example with the following structure:

```bash
forest-monitoring-example
├── data
│   ├── deployment
│   │   ├── client_1_demo_data.npz
│   │   └── client_2_demo_data.npz
│   ├── simulation
│   │   └── demo_data.npz
├── forest_monitoring_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
├── .gitignore
└── README.md
```

## Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

# Run the App
You can run your Flower App in both simulation and deployment mode without making changes to the code. If you are starting with Flower, we recommend you using the simulation mode as it requires fewer components to be launched manually. By default, flwr run will make use of the Simulation Engine.


## Run with the Simulation Engine

In the `forest-monitoring-example` directory, use `flwr run` to run a local simulation:

```bash
flwr run --stream
```

You can also override some of the settings for your ClientApp and ServerApp defined in pyproject.toml. For example:

```bash
flwr run . --run-config "num-server-rounds=3"
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.


```bash
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config=’client-id=”CID_1” data-path=”/path/to/demo_data/client_1_demo_data.npz”’
```


Launch the run via flwr run but pointing to a SuperLink connection that specifies the SuperLink your SuperNode is connected to:
```bash
flwr run . <SUPERLINK-CONNECTION> --stream
```

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Resources
This app is based on the preprint: https://zenodo.org/records/17415920

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
