---
tags: [quickstart, linear regression, tabular]
dataset: [Synthetic]
framework: [C++]
---

# Flower `SuperNodes` in C++

> [!WARNING]\
> This example is compatible with `flwr<1.13.0`. We are currently updating it to the newer `flwr run` way of running Flower Apps.

In this example you will train a linear model on synthetic data using C++ clients.

## Acknowledgements

Many thanks to the original contributors to this code:

- Lekang Jiang (original author and main contributor)
- Francisco José Solís (code re-organization)
- Andreea Zaharia (training algorithm and data generation)

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-cpp . \
        && rm -rf _tmp \
        && cd quickstart-cpp
```

This will create a new directory called `quickstart-cpp` with the following structure:

```shell
quickstart-cpp
├── py-server
|   ├── py_server
|   |   ├── server_app.py  # Defines your ServerApp
|   |   └── strategy.py    # Defines the strategy
|   └── pyproject.toml     # Defines python dependencies and ServerApp behaviour
├── include
│   └── *.h                # Various header files defining your client-side behaviour
├── src
│   └── *.cc               # Various source files defining your client-side behaviour
├── CMakeLists.txt         # Tells CMake how to build the C++ project
└── README.md
```

### Install dependencies and project

In this example, `SuperLink` and `ServerApp` use Flower's Python package, while the `SuperNodes` are C++ executables. We therefore need: (1) a Python environment with Flower installed; and (2) to build the `SuperNode` binaries.

> \[!NOTE\]
> Building the C++ components currently only works in Ubuntu 22.

1. **Prepare for `SuperLink` and `ServerApp`**

   In a new Python environment (Python 3.10.0 or higher), install the the project as defined in `pyproject.toml.` Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

   ```bash
   pip install -e py-server/
   ```

2. **Build the `SuperNode` executables**

   Ensure you have `CMake` and other build tools installed in your system.

   ```shell
   sudo apt-get install -y clang-format cmake g++ clang-tidy cppcheck
   ```

   Then, build the binaries:

   ```bash
   cmake -S . -B build
   cmake --build build -j 4 # -j allows you to parallelize your build
   ```

## Run the project

You will need 2+N terminals, where N>=2 and represents the number of `SuperNodes` that you connect to the `SuperLink`. First, launch the `SuperLink`

```bash
# From the python environemnt you created earlier
flower-superlink --insecure
```

Then, connect two `SuperNodes` indicating the address and port of the `SuperLink`. You'll notice the `SuperNodes` remain idle until you launch the `ServerApp`.

```bash
# Run on separate terminals
build/flower-supernode 0 127.0.0.1:9092
build/flower-supernode 1 127.0.0.1:9092
```

Finally, launch the `ServerApp`.

```bash
# From the python environment you created earlier
flower-server-app py-server --insecure
```
