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
    "flwr[simulation]>=1.9.0,<2.0",
    "flwr-datasets>=0.0.2,<1.0.0",
    "torch==2.2.1",
    "transformers>=4.30.0,<5.0"
    "evaluate>=0.4.0,<1.0"
    "datasets>=2.0.0, <3.0"
    "scikit-learn>=1.3.1, <2.0"
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[flower]
publisher = "$username"

[flower.components]
serverapp = "$import_name.server:app"
clientapp = "$import_name.client:app"

[flower.engine]
name = "simulation"

[flower.engine.simulation.supernode]
num = 2
