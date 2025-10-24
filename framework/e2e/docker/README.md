# Flower containers testing

This directory is used to test Flower containers in a minimum scenario, that is, with two `SuperNodes` connecting to a `SuperLink` without TLS. Both entities run in `isolation=subprocess`. The Flower app exectued is based on the `NumPy` template available via `flwr new`. This is mainly to test the functionalities of Flower in a containerized architecture. It can be easily extended to test more complex communication set-ups.

To execute locally, run the following in CLI:

```shell
# Pull the latest images
docker compose build --pull 
# Launch containers in the background
docker compose up -d
```

Submit a run:. It should take under 30s to complete

```shell
flwr run basic-app
```

To stop the containers, run:

```shell
$ docker compose down
```
