# =====================================================================
# For a full TOML configuration guide, check the Flower docs:
# https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html
# =====================================================================

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
# Dependencies for your Flower App
dependencies = [
    "flwr[simulation]>=1.23.0",
    "flwr-datasets>=0.5.0",
    "xgboost>=2.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "$username"

[tool.flwr.app.components]
serverapp = "$import_name.server_app:app"
clientapp = "$import_name.client_app:app"

# Custom config values accessible via `context.run_config`
[tool.flwr.app.config]
num-server-rounds = 3
fraction-train = 0.1
fraction-evaluate = 0.1
local-epochs = 1

# XGBoost parameters
params.objective = "binary:logistic"
params.eta = 0.1 # Learning rate
params.max-depth = 8
params.eval-metric = "auc"
params.nthread = 16
params.num-parallel-tree = 1
params.subsample = 1
params.tree-method = "hist"

# Default federation to use when running the app
[tool.flwr.federations]
default = "local-simulation"

# Local simulation federation with 10 virtual SuperNodes
[tool.flwr.federations.local-simulation]
options.num-supernodes = 10

# Remote federation example for use with SuperLink
[tool.flwr.federations.remote-federation]
address = "<SUPERLINK-ADDRESS>:<PORT>"
insecure = true  # Remove this line to enable TLS
# root-certificates = "<PATH/TO/ca.crt>"  # For TLS setup
