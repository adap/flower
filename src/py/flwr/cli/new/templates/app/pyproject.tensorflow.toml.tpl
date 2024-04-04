[build-system]
requires = ["poetry-core>=1.4.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "$project_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
authors = [
    "The Flower Authors <hello@flower.ai>",
]
readme = "README.md"

[tool.poetry.dependencies]
python = ">=3.9,<3.11"
# Mandatory dependencies
flwr = { version = "^1.8.0", extras = ["simulation"] }
flwr-datasets = { version = "^0.0.2", extras = ["vision"] }
tensorflow-cpu = { version = ">=2.9.1,<2.11.1 || >2.11.1", markers = "platform_machine == \"x86_64\"" }
tensorflow-macos = { version = ">=2.9.1,<2.11.1 || >2.11.1", markers = "sys_platform == \"darwin\" and platform_machine == \"arm64\"" }
