# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
ARG  BASE_VERSION=latest
FROM flwr/base:$BASE_VERSION as build

WORKDIR /app
RUN python -m pip install -U flwr[rest]
ENTRYPOINT ["python", "-c", "from flwr.server import run_server\nrun_server()"]


# Test if Flower can be successfully installed and imported
FROM build as test
RUN python -c "from flwr.server import run_server"
