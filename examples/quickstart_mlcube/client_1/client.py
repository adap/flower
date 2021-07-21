import os
import pickle
import flwr as fl
import subprocess
from typing import Dict, Any

MODULE_PATH = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(MODULE_PATH)
MLCUBE_DIR = os.path.join(MODULE_DIR, "mlcube")
MLCUBE_WORKSPACE_DIR = os.path.join(MLCUBE_DIR, "workspace")


def workspace_path(filepath: str) -> str:
    return os.path.join(MLCUBE_WORKSPACE_DIR, filepath)


def run_mlcube_task(task_name: str, output_file: str) -> Dict[str, Any]:
    """Run mlcube task and return if successful."""
    command = [
        "mlcube_docker",
        "run",
        "--mlcube=.",
        "--platform=platforms/docker.yaml",
        f"--task=run/{task_name}.yaml",
    ]

    subprocess.check_call(command, cwd=MLCUBE_DIR, stdout=subprocess.PIPE)

    print(f"Reading: {output_file}")
    results = pickle.load(open(output_file, "rb"))

    return results


def create_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    # Define Flower client
    class MLCubeClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            print("Calling MLCubeClient.get_parameters")
            output_file = workspace_path("get_parameters/output.p")
            create_directory(workspace_path("get_parameters"))

            results = run_mlcube_task("get_parameters", output_file)

            return results["parameters"]

        def fit(self, parameters, config):  # type: ignore
            print("Calling MLCubeClient.fit")
            input_file = workspace_path("fit/input.p")
            output_file = workspace_path("fit/output.p")
            create_directory(workspace_path("fit"))

            print(f"Writing: {input_file}")
            inputs = {"parameters": parameters, "config": config}
            pickle.dump(inputs, open(input_file, "wb"))

            print("Starting mlcube task fit")
            results = run_mlcube_task("fit", output_file)

            return results["parameters"], results["num_examples"], results["config"]

        def evaluate(self, parameters, config):  # type: ignore
            print("Calling MLCubeClient.evaluate")
            input_file = workspace_path("evaluate/input.p")
            output_file = workspace_path("evaluate/output.p")
            create_directory(workspace_path("evaluate"))

            print(f"Writing: {input_file}")
            inputs = {"parameters": parameters, "config": config}
            pickle.dump(inputs, open(input_file, "wb"))

            print("Starting mlcube task evaluate")
            results = run_mlcube_task("evaluate", output_file)

            return results["loss"], results["num_examples"], results["config"]

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=MLCubeClient())
