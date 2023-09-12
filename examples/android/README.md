<img src="https://developer.android.com/studio/images/studio-icon-preview.svg" alt="Android Logo" width="100">

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

### Installing Dependencies

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

# Run Federated Learning on Android Devices

Note : This example allows FLower to launch a long term & reliable continous Federated Learning Process on the Android devices.

(For Android Versions of 8 to 13)

The included `run.sh` will start the Flower server (using `server.py`). You can simply start it in a terminal as follows:

```shell
poetry run ./run.sh
```

Download and install the `flwr_android_client.apk` on each Android device/emulator. The server currently expects a minimum of 4 Android clients, but it can be changed in the `server.py`.

When the Android app runs for the first time, file access permission is asked to store the updates related to rounds participated in secondary storage of the phone. To initiate the Federated Learning process, App requires the client ID (between 1-10), the IP and port of your server.

Press the `Start` button to launch a background thread which will be responsible for loading the CIFAR10 dataset into memory, establishing the connection with server via GRPC Connection Channel , and contributing device in Federated Learning. A foreground service notification will appear to inidicate that Fedearted Learning process is ongoing in the background (some Android 13 sub-versions may not display the notification). Upon succesful updates such as loading dataset, connecting to server , or completing a round, App will display the progress under the heading `Result` subsequently.

Once the background thread is launched, the thread will manage the contribution of device, regardless of App state such as App in Background Tray or removed from it (except for Force stopping the app). Under the conditions such as network disconnectivity, device under memory presssure, or device rebooted, the background thread will relaunch itself persistently approxmiately after 15 minutes interval without any user intervention to recontribute. All the failures, updates and subsequent actions by the thread will be displayed under `Result`.

Press the `Stop` button to kill the background thread if you want to stop the device from contributing. And incase if you want to restart a new background thread, then re-enter all the feilds and press the start button.

The button `Refresh` will reload the latest updates by the background thread whereas the `Clear` button will remove all the contents of the file where the details of thread actions & progress are stored.

Incase, if you want to allow the background thread to run comfortably when the device display is off (in other words, device is in sleep mode) then press the button `Battery Optimisation` and then press allow to stop optimising the App. Without this permission, the thread will be given limited resource and may be killed by Andorid OS but it is guranteed to relaunch eventually to recontribute either during maintainnce window or when user turns on the device.

Please note that some Mobile Vendors have their own restrictions on the device which will prevent the thread to run even when `Battery Optimisation` is enabled.
