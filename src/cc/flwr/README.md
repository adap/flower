# Flower C++ Client SDK

You can install the C++ client SDK under this guidance. After successful installation, you can write a simple line "find_package(flwrLib REQUIRED)" to use our SDK to do federated learning in C++. An example can be found [here](https://github.com/adap/flower/tree/main/examples/quickstart_cpp).

# Install requirements

You'll need CMake

# Install guidance

```
mkdir build 
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
sudo cp -r ext/third_party/abseil-cpp/absl /usr/local/include
```

Now the Flower C++ client SDK has been installed!



