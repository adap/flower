# Flower

# Setup
For the initial setup run the following
```bash
$ git clone git@github.com:adap/flower.git
$ ./dev/bootstrap.sh
```

## Compile proto definitions
In case the you change the proto definitions you will have to recompile.
Do do that run:
```bash
$ python -m flower_tools.grpc
```
