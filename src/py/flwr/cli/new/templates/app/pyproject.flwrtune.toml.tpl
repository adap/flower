[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
authors = [
    { name = "The Flower Authors", email = "hello@flower.ai" },
]
license = { text = "Apache License (2.0)" }
dependencies = [
    "flwr[simulation]>=1.8.0,<2.0",
    "flwr-datasets>=0.1.0,<1.0.0",
    "hydra-core==1.3.2",
    "trl==0.8.1",
    "bitsandbytes==0.43.1",
    "scipy==1.13.0",
    "peft==0.6.2",
    "transformers==4.39.3",
    "sentencepiece==0.2.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[flower]
publisher = "$username"

[flower.components]
serverapp = "$import_name.app:server"
clientapp = "$import_name.app:client"

[flower.engine]
name = "simulation"

[flower.engine.simulation.supernode]
num = $num_clients

[flower.engine.simulation]
backend_config = { client_resources = { num_cpus = 8, num_gpus = 1.0 } }
