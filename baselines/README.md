# Flower Baselines


> Previous baselines can still be found in `flwr_baselines`. These will be gradually migrated to the new directory structure.

## Running the baselines

Each basline is self-contained in its own directory. Furthermore, each basline defines its own Python environment using [Poetry](https://python-poetry.org/docs/) via a `pyproject.toml` file. In order to run a baseline:

1. Navigate inside the directory of the baseline you'd like to run
2. Follow the `[Environment Setup]` instructions in the `README.md`. In most cases this will require you to just do:

    ```bash
    poetry install
    ```
3. Run the baseline as indicated in the `[Running the Experiments]` section in the `README.md`


## Contributing a new baseline

Do you have a new federated learning paper and want to add a new baseline to Flower? Or do you want to add an experiment to an existing baseline paper? Great, we really appreciate your contribution !!

The steps to follow are:

1. Fork the Flower repo and clone it into your machine
2. Navigate to the `baselines/` directory and from there run:
    ```bash
    # This will create a new directory with the same structure as `baseline_template`.
    ./dev/creat-_baseline.sh <your_baseline_name>
    ``` 
3. Ensure you follow the step showing after running the script above. This will ensure that a Python project is properly constructed for your baseline.
4. Then, go inside your basline directory and continue with the steps detailed in `EXTENDED_README.md` and `README.md`.
