[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.10.0",
    "flwr-datasets>=0.3.0",
    "xgboost>=2.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "$username"

[tool.flwr.app.components]
serverapp = "$import_name.server_app:app"
clientapp = "$import_name.client_app:app"

[tool.flwr.app.config]
# ServerApp
num-server-rounds = 3
fraction-fit = 0.1
fraction-evaluate = 0.1

# ClientApp
local-epochs = 1
params.objective = "binary:logistic"
params.eta = 0.1  # Learning rate
params.max-depth = 8
params.eval-metric = "auc"
params.nthread = 16
params.num-parallel-tree = 1
params.subsample = 1
params.tree-method = "hist"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 20
