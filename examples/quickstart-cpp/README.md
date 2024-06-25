---
title: Simple Flower Example using C++
url: https://pytorch.org/
labels: [quickstart, linear regression, tabular]
dataset: [synthetic]
framework: [c++]
---

# Flower Clients in C++ (under development)

In this example you will train a linear model on synthetic data using C++ clients.

## Acknowledgements

Many thanks to the original contributors to this code:

- Lekang Jiang (original author and main contributor)
- Francisco José Solís (code re-organization)
- Andreea Zaharia (training algorithm and data generation)

## Install requirements

You'll need CMake and Python with `flwr` installed.

### Building the example

This example provides you with a `CMakeLists.txt` file to configure and build the client. Feel free to take a look inside it to see what is happening under the hood.

```bash
cmake -S . -B build
cmake --build build
```

## Run the `Flower SuperLink`, the two clients, and the `Flower ServerApp` in separate terminals

```bash
flwr-superlink --insecure
```

```bash
build/flwr_client 0 127.0.0.1:9092
```

```bash
build/flwr_client 1 127.0.0.1:9092
```

```bash
flower-server-app server:app --insecure
```
