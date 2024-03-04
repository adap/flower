[project]
name = "$project_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
authors = ["The Flower Authors <hello@flower.ai>"]

[flower.components]
serverapp = "$project_name.server:app"
clientapp = "$project_name.client:app"

[flower.engine]
name = "simulation" # optional

[flower.engine.simulation.super-node]
count = 10 # optional
