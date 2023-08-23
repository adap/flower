# Secure Aggregation with Driver API

This example contains highly experimental code. Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart_pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced_pytorch)) to learn how to use Flower with PyTorch.

## Installing Dependencies

Project dependencies (such as and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

## Run with Driver API

```bash
./run.sh
```
