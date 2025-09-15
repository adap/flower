"""huggingface_example: A Flower / Hugging Face app."""

from flwr.app import Context, ArrayRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from huggingface_example.task import get_model


app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    # Define model to federate and extract parameters
    model_name = context.run_config["model-name"]
    model = get_model(model_name)
    arrays = ArrayRecord(model.state_dict())
    
    # Instantiate strategy
    fraction_train = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )
    
    num_rounds = context.run_config["num-server-rounds"]
    # Start the strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
