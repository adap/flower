
# Extended Readme

> The baselines are expected to run in a machine running Ubuntu

While `README.md` should include information about the baseline you implement and how to run it, this _extended_ readme provides info on what's the expected directory structure for a new baseline. Please follow closely these instructions. 

1. For the Flower repository.
2. Clone the `basline_template` directory and rename it with a name that makes your baseline easily identifiable.
3. Also rename with the same name the sub-directory that's by default assigned the `baseline_template` name.
3. All your code and configs should go into a sub-directory with the same name as the name of your baseline.
    *    The sub-directory contains a series of Python scripts that you can edit. Please stick to these files and consult with us if you need additional ones.
    *    There is also a basic config structure in `<your_baseline_name>/conf` ready be parsed by [Hydra](https://hydra.cc/) when exectuing `<your_baseline_name>/main.py`.
4. Therefore, the top-level directory should only include:
    *    A directory where all your code+configs live
    *    A `README.md` describing your baseline. Plese follow the template in the provided `README.md`
    *    A `pyproject.toml` detailing the Python environment construction process via [Poetry](https://python-poetry.org/docs/)
        *    Make sure the variable `name` in `pyproject.toml` is set to the name of the sub-directory containing all your code.
5. Add your dependencies to the `pyproject.toml` (See below a few examples on how to do it)
6. Ensure that the Python environment for your basline can be created without errors by simply running: `poetry install`
7. Ensure that your baseline runs with default argument by running `poetry run python -m <your_baseline_name>/main.py`. Then, follow the instructions provided in the `README.md` and detail the steps to follow in `Environment Setup` and in `Running the Experiments`.

> Once you are happy to merge your baseline contribution, please delete this `EXTENDED_README.md` file.


## About Poetry

We use Poetry to manage the Python environment for each individual baseline. You can follow the instructions [here](https://python-poetry.org/docs/) to installl Poetry in your machine. 

With Poetry already installed, you can create an environment for this baseline with commands:
```bash
# run this from the same directory as the `pyproject.toml` file is
poetry install
```

This will create a basic Python environment with just Flower and additional packages, including those needed for simulation. Next, you should add the dependencies for your code. It is **critical** that you fix the version of the packages you use using a `=` not a `=^`. You can so via [`poetry add`](https://python-poetry.org/docs/cli/#add):

```bash
# For instance, if you want to install tqdm
poetry add tqdm==4.65.0

# If you already have a requirements.txt, you can add all those packages (but ensure you have fixed the version) in one go as follows:
poetry add $( cat requirements.txt )
```

More critically however, is adding your ML framework of choice to the list of dependencies. 

TODO