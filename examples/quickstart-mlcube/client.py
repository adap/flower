import os
import sys
import flwr as fl
import mlcube_utils as mlcube


# Define Flower client
class MLCubeClient(fl.client.NumPyClient):
    def __init__(
        self, workspace: str, optimizer="adam", epochs=1, batch_size=32
    ) -> None:
        super().__init__()
        self.workspace = workspace

        mlcube.write_hyperparameters(self.workspace, optimizer, epochs, batch_size)
        mlcube.run_task(self.workspace, "download")

    def get_parameters(self, config):  # type: ignore
        pass

    def fit(self, parameters, config):  # type: ignore
        mlcube.save_parameteras_as_model(self.workspace, parameters)
        mlcube.run_task(self.workspace, "train")
        parameters = mlcube.load_model_parameters(self.workspace)
        config = mlcube.load_train_metrics(self.workspace)
        return parameters, config["num_examples"], config

    def evaluate(self, parameters, config):  # type: ignore
        mlcube.save_parameteras_as_model(self.workspace, parameters)
        mlcube.run_task(self.workspace, "evaluate")
        config = mlcube.load_evaluate_metrics(self.workspace)
        return config["loss"], config["num_examples"], config


def main():
    """Start Flower client.

    Use first argument passed as workspace name
    """
    workspace_name = sys.argv[1]

    workspace = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "workspaces", workspace_name
    )

    fl.client.start_client(
        server_address="0.0.0.0:8080",
        client=MLCubeClient(workspace=workspace).to_client(),
    )


if __name__ == "__main__":
    main()
