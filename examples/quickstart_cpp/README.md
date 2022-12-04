# Flower Clients in C++

In this example you will train a linear model on synthetic data using C++ clients.

# Acknowledgements 
Many thanks to the original contributors to this code:
- Lekang Jiang (original author and main contributor)
- Francisco José Solís (code re-organization)
- Andreea Zaharia (training algorithm and data generation)

# Install requirements
You'll need CMake and Python. Make sure you have installed [C++ SDK](https://github.com/adap/flower/tree/main/src/cc/flwr) successfully. 

### Building the example
This example provides you with a `CMakeLists.txt` file to configure and build the client. Feel free to take a look inside it to see what is happening under the hood.


```bash
mkdir build
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

# How to use C++ SDK

In CMakeLists.txt

```
find_package(flwrLib REQUIRED)

target_include_directories(${EXECUTABLE_NAME}
  ${FLWR_INCLUDE_DIRS}
)

target_link_libraries(${EXECUTABLE_NAME}
  ${FLWR_LIBRARIES}
)
```

In your code

```c++
#include "flwr_lib/client.h"
#include "flwr_lib/start.h"
```

# Useful hints

If you meet problems like "cannot open shared object file", try the following line. 

```
sudo /sbin/ldconfig -v
```

