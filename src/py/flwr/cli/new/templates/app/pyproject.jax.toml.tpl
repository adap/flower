[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.17.0",
    "jax==0.4.30",
    "jaxlib==0.4.30",
    "scikit-learn==1.6.1",
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
input-dim = 3

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
