import os
import argparse
import flwr as fl
import tensorflow as tf
import logging
from helpers.load_data import load_data
import os
from model.model import Model

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Flower client')

parser.add_argument('--server_address', type=str, default="server:8080")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.1)

args = parser.parse_args()

# Create an instance of the model and pass the learning rate as an argument
model = Model(learning_rate=args.learning_rate)

# Compile the model
model.compile()

class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

    def get_parameters(self, config):
        # Return the parameters of the model
        return model.get_model().get_weights()


    def fit(self, parameters, config):

        # Set the weights of the model
        model.get_model().set_weights(parameters)

        # Load the training dataset and get the number of examples
        train_dataset, _, num_examples_train, _ = load_data(batch_size=self.args.batch_size)
        
        # Train the model
        history = model.get_model().fit(train_dataset)

        # Calculate evaluation metric
        results = {
            "accuracy": float(history.history["accuracy"][-1]),
        }         

        # Get the parameters after training
        parameters_prime = model.get_model().get_weights()
       
        # Directly return the parameters and the number of examples trained on
        return parameters_prime, num_examples_train, results


    
    def evaluate(self, parameters, config):
        
        # Set the weights of the model
        model.get_model().set_weights(parameters)

        # Use the test dataset for evaluation
        _, test_dataset, _, num_examples_test = load_data(batch_size=self.args.batch_size)

        # Evaluate the model and get the loss and accuracy
        loss, accuracy = model.get_model().evaluate(test_dataset)

        # Return the loss, the number of examples evaluated on and the accuracy
        return float(loss), num_examples_test, {"accuracy": float(accuracy)}
    

# Function to Start the Client
def start_fl_client():
    try:
        fl.client.start_numpy_client(server_address=args.server_address, client=Client(args))
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()