[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.9.0,<2.0",
    "flwr-datasets>=0.0.2,<1.0.0",
    "xgboost>=2.0.0,<3.0.0",
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
num-server-rounds = "3"
pool-size = "2"
num-clients-per-round = "2"
num-evaluate-clients = "2"

# ClientApp
local-epochs = "1"
lr = "0.1"
max-depth = "8"

[tool.flwr.federations]
default = "localhost"

[tool.flwr.federations.localhost]
options.num-supernodes = 10
