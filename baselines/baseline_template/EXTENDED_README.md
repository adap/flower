
# Extended Readme

> The baselines are expected to run in a machine running Ubuntu 22.04

While `README.md` should include information about the baseline you implement and how to run it, this _extended_ readme provides info on what's the expected directory structure for a new baseline. Please follow closely these instructions. It is likely that you have already completed steps 1-3.

1. Fork the Flower repository and clone it.
2. Navigate to the `baselines/` directory and from there run:
    ```bash
    # This will create a new directory with the same structure as this `baseline_template` directory.
    ./dev/create-baseline.sh <your_baseline_name>
    ``` 
3. After running the command above, you'll be asked to rename the `name` field in the `pyproject.toml` using `your_baseline_name`.
4. All your code and configs should go into a sub-directory with the same name as the name of your baseline.
    *    The sub-directory contains a series of Python scripts that you can edit. Please stick to these files and consult with us if you need additional ones.
    *    There is also a basic config structure in `<your_baseline_name>/conf` ready be parsed by [Hydra](https://hydra.cc/) when exectuing `<your_baseline_name>/main.py`.
5. Therefore, the top-level directory should only include:
    *    A directory where all your code+configs live
    *    A `README.md` describing your baseline. Please follow the template in the provided `README.md`
    *    A `pyproject.toml` detailing the Python environment construction process via [Poetry](https://python-poetry.org/docs/)
        *    Make sure the variable `name` in `pyproject.toml` is set to the name of the sub-directory containing all your code.
6. Add your dependencies to the `pyproject.toml` (See below a few examples on how to do it)
7. Ensure that the Python environment for your baseline can be created without errors by simply running: `poetry install`
8. Ensure that your baseline runs with default argument by running `poetry run python -m <your_baseline_name>/main`. Then, follow the instructions provided in the `README.md` and detail the steps to follow in `Environment Setup` and in `Running the Experiments`.
9. Once you have your code is ready and you have checked:
    *    that following the instructions in your `README.md` the Python environment can be created correctly

    *    that running the code following your instructions can reproduce the experiments in the paper
   
   , then you just need to create a Pull Request (PR) to kickstart the process of merging your baseline into the Flower repository.

> Once you are happy to merge your baseline contribution, please delete this `EXTENDED_README.md` file.


## About Poetry

We use Poetry to manage the Python environment for each individual baseline. You can follow the instructions [here](https://python-poetry.org/docs/) to install Poetry in your machine. 

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
With each `poetry add` command, the `pyproject.toml` gets automatically updated so you don't need to keep that `requirements.txt` as part of this baseline.


More critically however, is adding your ML framework of choice to the list of dependencies. For some frameworks you might be able to do so with the `poetry add` command. Check [the Poetry documentation](https://python-poetry.org/docs/cli/#add) for how add packages in various ways. For instance, let's say you want to use PyTorch:

```bash
# with plain `pip`  you'd run a command such as:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# to add the same 3 dependencies to your Poetry environment you'd need to add the URL to the wheel that the above pip command auto-resolves for you. You can find those wheels in `https://download.pytorch.org/whl/cu117`. Copy the link and paste it after the `poetry add` command.
# For instance to add `torch==1.13.1+cu117` and a x86 Linux system with Python3.8 you'd:
poetry add https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl
# you'll need to repeat this for both `torchvision` and `torchaudio`
```
The above is just an example of how you can add these dependencies. Please refer to the Poetry documentation to extra reference.

If all attempts fail, you can still install packages via standard `pip`. You'd first need to source/activate your Poetry environment.
```bash
# first ensure you have created your environment
poetry install 

# then activate it
poetry shell
```
Now you are inside your environment (pretty much as when you use `virtualenv` or `conda`) so you can install further packages with `pip`. Please note that, unlike with `poetry add`, these extra requirements won't be captured by `pyproject.toml`. Therefore, please ensure that you provide all instructions needed to: (1) create the base environment with Poetry and (2) install any additional dependencies via `pip`.