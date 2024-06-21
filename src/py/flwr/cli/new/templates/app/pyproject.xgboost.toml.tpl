[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$project_name"
version = "1.0.0"
description = ""
authors = [
    { name = "The Flower Authors", email = "hello@flower.ai" },
]
license = {text = "Apache License (2.0)"}
dependencies = [
    "flwr[simulation]>=1.8.0,<2.0",
    "flwr-datasets>=0.0.2,<1.0.0",
    "xgboost>=2.0.0,<3.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[flower.components]
serverapp = "$project_name.server:app"
clientapp = "$project_name.client:app"
