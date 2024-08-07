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
    "flwr-datasets>=0.1.0,<1.0.0",
    "hydra-core==1.3.2",
    "trl==0.8.1",
    "bitsandbytes==0.43.0",
    "scipy==1.13.0",
    "peft==0.6.2",
    "transformers==4.39.3",
    "sentencepiece==0.2.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "$username"

[tool.flwr.app.components]
serverapp = "$import_name.app:server"
clientapp = "$import_name.app:client"

[tool.flwr.app.config]
num-server-rounds = 3

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
