import flwr as fl
from driver.server import Server
from flwr.server.client_manager import SimpleClientManager


def gen_config_fn(model_config, tflite_bytes=None):
    def config_fn(rnd):
        config = model_config
        if rnd == 1 and tflite_bytes:
            config["tf_lite"] = tflite_bytes
        return config

    return config_fn


def run_driver(builder, new_model=True, py_client=False, address="65.108.122.72"):
    model_config = builder.config
    initial_params = builder.initial_parameters if new_model else None
    tflite_model = builder.tflite_model if new_model else None

    strategy = (
        fl.server.strategy.FedAvg if py_client else fl.server.strategy.FedAvgAndroid
    )
    server = Server(
        client_manager=SimpleClientManager(),
        model_id=model_config["model_id"],
        strategy=strategy(
            initial_parameters=initial_params,
            on_evaluate_config_fn=gen_config_fn(model_config),
            on_fit_config_fn=gen_config_fn(model_config, tflite_model),
            min_available_clients=2,
            min_fit_clients=2,
            min_evaluate_clients=2,
        ),
    )
    fl.driver.start_driver(
        server_address=f"{address}:9091",
        config=fl.server.ServerConfig(num_rounds=3),
        server=server,
    )
