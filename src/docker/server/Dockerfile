# Copyright 2023 Flower Labs GmbH. All Rights Reserved.

ARG BASE_VERSION=22.04
FROM flwr/base:$BASE_VERSION

WORKDIR /app
ARG FLWR_VERSION
RUN python -m pip install -U --no-cache-dir flwr[rest]==${FLWR_VERSION}
ENTRYPOINT ["python", "-c", "from flwr.server import run_server; run_server()"]

# Test if Flower can be successfully installed and imported
FROM flwr/base:$BASE_VERSION
RUN python -c "from flwr.server import run_server"
