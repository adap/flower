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
    "scikit-learn>=1.1.1",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[flower]
publisher = "$username"

[flower.components]
serverapp = "$import_name.server:app"
clientapp = "$import_name.client:app"
