[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.13.1",
    "flwr-datasets[vision]>=0.3.0",
    "tensorflow>=2.11.1,<2.18.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "$username"

[tool.flwr.app.components]
serverapp = "$import_name.server_app:app"
clientapp = "$import_name.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 3
local-epochs = 1
batch-size = 32
verbose = false

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
