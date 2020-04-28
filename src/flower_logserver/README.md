# Flower LogServer
A simple server which receives logs from the python standard library `logging.handlers.HTTPHandler`
and prints them to the console.

# Quickstart
Just run:
```bash
# A very minimal example showcasing how centrailized logging works
# Run these commands in 3 different terminals

# Start the logserver
python -m flower_logserver

# Start the FL server while ignoring stdout/stderr
FLOWER_LOG_HTTP=localhost:8081 python -m flower_benchmark.tf_fashion_mnist.server

# Start a FL client while ignoring stdout/stderr
FLOWER_LOG_HTTP=localhost:8081 python -m flower_benchmark.tf_fashion_mnist.client \
    --cid=0 --partition=0 --clients=1 --grpc_server_address=localhost --grpc_server_port=8080
```