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
    "torch==2.2.1",
    "transformers>=4.30.0,<5.0",
    "evaluate>=0.4.0,<1.0",
    "datasets>=2.0.0, <3.0",
    "scikit-learn>=1.3.1, <2.0",
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

[tool.flwr.federations]
default = "localhost"

[tool.flwr.federations.localhost]
options.num-supernodes = 10
