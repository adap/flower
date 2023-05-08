# Flower Clients in C++

In this example you will train a linear model on synthetic data using C++ clients.

# Acknowledgements 
Many thanks to the original contributors to this code:
- Lekang Jiang (original author and main contributor)
- Francisco José Solís (code re-organization)
- Andreea Zaharia (training algorithm and data generation)

# Install requirements
You'll need CMake and Python.

### Building the example
This example provides you with a `CMakeLists.txt` file to configure and build the client. Feel free to take a look inside it to see what is happening under the hood.


```bash
cmake -S . -B build 
cmake --build build
```

# Run the server and two clients in separate terminals
```bash 
python server.py
```
```bash
build/flwr_client 0 127.0.0.1:8080 
```
```bash
build/flwr_client 1 127.0.0.1:8080
```
