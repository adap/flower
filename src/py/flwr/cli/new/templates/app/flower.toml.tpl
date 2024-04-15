[project]
name = "$project_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
authors = [
    "The Flower Authors <hello@flower.ai>",
]
readme = "README.md"

[flower.components]
serverapp = "$project_name.server:app"
clientapp = "$project_name.client:app"
