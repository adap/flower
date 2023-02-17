# Flower iOS - A Flower SDK for iOS Devices with Example

Flower iOS contains a Swift Package for Flower clients in iOS and an example that demonstrates a federated learning setup with iOS clients. The training is done on a MNIST dataset using CoreML with a updatable digit recognition model.

## Project Setup

Project dependencies (`flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run Federated Learning on iOS Clients

To start the server, write the following command in the terminal in the ios folder (with the conda environment created above):

```shell
python3 server.py
```

Open the FlowerCoreML.xcodeproj with XCode, wait until the dependencies are fetched, then click build and run with iPhone 13 Pro Max as target, or you can deploy it in your own iOS device by connecting your Mac, where you run your XCode, and your iPhone.

When the iOS app runs, load both the training and test dataset first. Then enter the hostname and port of your server in the TextField provided. Finally press `Start` which will start the federated training.
