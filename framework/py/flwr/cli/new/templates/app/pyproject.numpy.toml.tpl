[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
# These are the dependencies for your FlowerApp
# The Python packages listed here will be installed
# when you do `pip install -e .`
dependencies = [
    "flwr[simulation]>=1.20.0",
    "numpy>=2.0.2",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "$username"

# The `serverapp` and `clientapp` items point
# to the `ServerApp` and `ClientApp` objects defined
# in your FlowerApp.
[tool.flwr.app.components]
serverapp = "$import_name.server_app:app"
clientapp = "$import_name.client_app:app"

# This section can include hyperparameters or configuration
# values relevant to your ClientApp and ServerApp. Note that
# the types supported are limited by the TOML syntaxt.
# Both the ClientApp and SeverApp can load these values at
# runtime through the context. In your app do:
#
#   num_server_rounds = context.run_config["num-server-rounds"]
#
# You can as many config values as your FlowerApp needs and access
# them as shown above via the Context.
[tool.flwr.app.config]
num-server-rounds = 3

# Indicating what your default federation is useful so you
# don't need to set it when you use the Flower CLI commands
[tool.flwr.federations]
default = "local-simulation"

# This is a federation for simulation named "local-simulation"
# Note that the name of the federation is arbitrary (you can 
# set it to anyother name)
# Learn more how to customize your simulation from the docs:
# https://flower.ai/docs/framework/how-to-run-simulations.html
[tool.flwr.federations.local-simulation]
options.num-supernodes = 10 # This federation defines 10 SuperNodes. This means that the 
                            # Flower Simulation Runtime will use 10 virtual nodes.

# To run your FlowerApp using the Flower Deployment Runtime
# (https://flower.ai/docs/framework/deploy.html)
# you need to set 
[tool.flwr.federations.remote-federation]
address = "<SUPERLINK-ADDRESS>:9093"     # Address of the SuperLink Exec API
insecure = true                          # Remove this if you want to run with TLS and specify `root-certificates`
# root-certificates = "<PATH/TO/ca.crt>" # TLS certificate set for the SuperLink. Required for self-signed certificates.
