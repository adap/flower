# Flower Client with LibTorch (C++)

In this example you will train a ResNet18 network on CIFAR10 using Libtorch (C++) clients.

Dataset download and other parts of this code are based on [PyTorch-CPP](https://github.com/prabhuomkar/pytorch-cpp).

# Install requirements
Before we begin, make sure you have both LibTorch and TorchVision C++ libraries installed on your systems.
- [LibTorch](https://pytorch.org/get-started/locally/)
- [TorchVision](https://github.com/pytorch/vision#using-the-models-on-c)


### Building 
This example provides you with a `CMakeLists.txt` file to configure and build the client. Feel free to take a look inside it to see what is happening under the hood.

To build the project, run the code below and append `-D DOWNLOAD_DATASETS=ON` in case you need to download the CIFAR10 dataset. 
Remember to change the placeholders `ABSOLUTE_PATH_TO_LIBTORCH` and `ABSOLUTE_PATH_TO_TORCHVISION` to absolute paths to the **folders** containing the libraries before running the commands below:


```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="ABSOLUTE_PATH_TO_LIBTORCH;ABSOLUTE_PATH_TO_TORCHVISION" -D DOWNLOAD_DATASETS=ON
cmake --build build
```

# Run the server and two clients in separate terminals
```bash 
python server.py
```

```bash
build/client 0 [::]:8080 /home/user/flower/examples/libtorch_client/data/cifar10/
```