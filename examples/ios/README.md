# Flower iOS - A Flower SDK for iOS Devices with Example
Flower iOS contains a Swift Package for Flower clients in iOS and an example that demonstrates a federated learning setup with iOS clients. The training is done on a MNIST dataset using CoreML with a updatable digit recognition model. 

## Project Setup

Requirements for setting up a server:
- Conda for managing environment
- Python3

Requirements for setting up a client:
- XCode

To set up the project, start by cloning the ios folder. After that create a new conda environment.

```shell
conda create --name flower-ios
```

Activate the created conda enviroment

```shell
conda activate flower-ios
```

Install flwr using pip3

```shell
pip3 install flwr
```

# Run Federated Learning on iOS Clients

To start the server, write the following command in the terminal in the ios folder (with the conda environment created above):

```shell
python3 run server.py
```

Open the FlowerCoreML.xcodeproj with XCode, wait until the dependencies are fetched, then click build and run with iPhone 13 Pro Max as target, or you can deploy it in your own iOS device by connecting your Mac, where you run your XCode, and your iPhone.

When the iOS app runs, load both the training and test dataset first. Then enter the hostname and port of your server in the TextField provided. Finally press `Start` which will start the federated training.
