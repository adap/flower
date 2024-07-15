[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = { text = "Apache License (2.0)" }
dependencies = [
    "flwr[simulation]>=1.9.0,<2.0",
    "flwr-datasets[vision]>=0.0.2,<1.0.0",
    "mlx==0.10.0",
    "numpy==1.24.4",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr]
publisher = "$username"

[tool.flwr.components]
serverapp = "$import_name.server:app"
clientapp = "$import_name.client:app"

[tool.flwr.federations]
default = "localhost"

[tool.flwr.federations.localhost]
options.num-supernodes = 10
