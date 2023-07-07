# Flower Android Client Example with Kotlin and TensorFlow Lite 2022

## Set up

Download the training and testing data from <https://www.dropbox.com/s/coeixr4kh8ljw6o/cifar10.zip?dl=1> and extract them to `client/app/src/main/assets/data`.

Download the TFLite model from <https://github.com/FedCampus/dyn_flower_android_drf/files/11858642/cifar10.zip> to `client/app/src/main/assets/model/cifar10.tflite`.
Alternatively, see `gen_tflite/README.md` for information on how to convert the CIFAR10 models to a `.tflite` file.

Download Python dependencies:

```sh
python3 -m pip install -r requirements.txt
```

## Run the demo

Start the Flower server at `./`:

```sh
python3 server.py
```

Install the app on *physical* Android devices and launch it.

*Note*: the highest tested JDK version the app supports is 16; it fails to build using JDK 19 on macOS.

In the user interface, fill in:

- Device number: a unique number among 1 ~ 10.
  This number is used to choose the partition of the training dataset.
- Server IP: an IPv4 address of the computer your backend server is running on. You can probably find it in your system network settings.
- Server port: 8080.

Push the first button and load the dataset. This may take a minute.

Push the second button and start the training.

______________________________________________________________________

Note: don't follow the instructions below. They are left here for reference only.

Original README from Flower Android example:

# Flower Android Example (TensorFlowLite)

This example demonstrates a federated learning setup with Android Clients. The training on Android is done on a CIFAR10 dataset using TensorFlow Lite. The setup is as follows:

- The CIFAR10 dataset is randomly split across 10 clients. Each Android client holds a local dataset of 5000 training examples and 1000 test examples.
- The FL server runs in Python but all the clients run on Android.
- We use a strategy called FedAvgAndroid for this example.
- The strategy is vanilla FedAvg with a custom serialization and deserialization to handle the Bytebuffers sent from Android clients to Python server.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/android . && rm -rf flower && cd android
```

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run Federated Learning on Android Devices

The included `run.sh` will start the Flower server (using `server.py`). You can simply start it in a terminal as follows:

```shell
poetry run ./run.sh
```

Download and install the `flwr_android_client.apk` on each Android device/emulator. The server currently expects a minimum of 4 Android clients, but it can be changed in the `server.py`.

When the Android app runs, add the client ID (between 1-10), the IP and port of your server, and press `Load Dataset`. This will load the local CIFAR10 dataset in memory. Then press `Setup Connection Channel` which will establish connection with the server. Finally, press `Train Federated!` which will start the federated training.
