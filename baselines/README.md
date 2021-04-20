# Flower Baselines

## Project Setup

Start by cloning the Flower Baselines project. We prepared a single-line command that you can copy into your shell which will clone the project for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/baselines . && rm -rf flower && cd baselines
```

Project dependencies (such as `flwr` and `torch`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr_baselines"
```

### TLDR

There is also a short way which might not be suitable for everyone.

```shell
./dev/bootstrap.sh
```
