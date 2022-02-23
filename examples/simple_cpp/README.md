# Simple Flower Client using C++

In this example you will train a linear model on synthetic data using C++ clients.

# Acknowledgements 
Many thanks to the original contributors to this code:
- Lekang Jiang (original author and main contributor)
- Francisco José Solís

# Install requirements
TODO

### Building 
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
build/client 0 [::]:8080 
```