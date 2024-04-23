[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$module_name"
version = "1.0.0"
description = ""
authors = [
    { name = "The Flower Authors", email = "hello@flower.ai" },
]
license = {text = "Apache License (2.0)"}
dependencies = [
    "flwr[simulation]>=1.8.0,<2.0",
    "numpy>=1.21.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[flower.components]
serverapp = "$module_name.server:app"
clientapp = "$module_name.client:app"
