# Flower LogServer
A simple server which receives logs from the python standard library `logging.handlers.HTTPHandler` and prints them to the console.

# Quickstart
A minimal example showing how centralized logging works.

Run these commands in 3 different terminals.
Start the log server.
```bash
python -m flower_logserver
```

Start the FL server and client.
```bash
python -m flower_benchmark.tf_fashion_mnist.server --log_host=localhost:8081
```

```bash
python -m flower_benchmark.tf_fashion_mnist.client \
    --cid=0 --partition=0 --clients=1 --grpc_server_address=localhost --grpc_server_port=8080 \
    --log_host=localhost:8081
```

# Persist logs to S3
If you would like to upload your logs regularly to S3 you can pass the following command line arguments on start.
```bash
python -m flower_logserver --s3_bucket=MY_BUCKET --s3_key=MY_S3_KEY
```
