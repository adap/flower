[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.13.0",
    "flwr-datasets[vision]>=0.3.0",
    "torch==2.2.1",
    "torchvision==0.17.1",
]

[tool.hatch.metadata]
allow-direct-references = true

[project.optional-dependencies]
dev = [
    "isort==5.13.2",
    "black==24.2.0",
    "docformatter==1.7.5",
    "mypy==1.8.0",
    "pylint==3.2.6",
    "flake8==5.0.4",
    "pytest==6.2.4",
    "pytest-watch==4.2.0",
    "ruff==0.1.9",
    "types-requests==2.31.0.20240125",
]

[tool.isort]
profile = "black"
known_first_party = ["flwr"]

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311"]

[tool.pytest.ini_options]
minversion = "6.2"
addopts = "-qq"
testpaths = [
    "flwr_baselines",
]

[tool.mypy]
ignore_missing_imports = true
strict = false
plugins = "numpy.typing.mypy_plugin"

[tool.pylint."MESSAGES CONTROL"]
disable = "duplicate-code,too-few-public-methods,useless-import-alias"
good-names = "i,j,k,_,x,y,X,Y,K,N"
max-args = 10
max-attributes = 15
max-locals = 36
max-branches = 20
max-statements = 55

[tool.pylint.typecheck]
generated-members = "numpy.*, torch.*, tensorflow.*"

[[tool.mypy.overrides]]
module = [
    "importlib.metadata.*",
    "importlib_metadata.*",
]
follow_imports = "skip"
follow_imports_for_stubs = true
disallow_untyped_calls = false

[[tool.mypy.overrides]]
module = "torch.*"
follow_imports = "skip"
follow_imports_for_stubs = true

[tool.docformatter]
wrap-summaries = 88
wrap-descriptions = 88

[tool.ruff]
target-version = "py38"
line-length = 88
select = ["D", "E", "F", "W", "B", "ISC", "C4"]
fixable = ["D", "E", "F", "W", "B", "ISC", "C4"]
ignore = ["B024", "B027"]
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".pytype",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
    "proto",
]

[tool.ruff.pydocstyle]
convention = "numpy"

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "$username"

[tool.flwr.app.components]
serverapp = "$import_name.server_app:app"
clientapp = "$import_name.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 3
fraction-fit = 0.5
local-epochs = 1

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.0
