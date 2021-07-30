import flwr as fl
import mlcube_utils as mlcube


# Define Flower client
class MLCubeClient(fl.client.NumPyClient):
    def __init__(self, optimizer="adam", epochs=1, batch_size=32) -> None:
        super().__init__()
        mlcube.write_hyperparameters(optimizer, epochs, batch_size)
        mlcube.run_task("download")

    def get_parameters(self):  # type: ignore
        pass

    def fit(self, parameters, config):  # type: ignore
        mlcube.save_parameteras_as_model(parameters)
        mlcube.run_task("train")
        parameters = mlcube.load_model_parameters()
        config = mlcube.load_train_metrics()
        return parameters, config["num_examples"], config

    def evaluate(self, parameters, config):  # type: ignore
        mlcube.save_parameteras_as_model(parameters)
        mlcube.run_task("evaluate")
        config = mlcube.load_evaluate_metrics()
        return config["loss"], config["num_examples"], config


if __name__ == "__main__":
    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=MLCubeClient())
