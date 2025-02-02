"""Define the Flower Client and function to instantiate it."""

import math

import numpy as np
from hydra.utils import instantiate
from keras.utils import to_categorical

import flwr as fl


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client."""

    # pylint: disable=too-many-arguments
    def __init__(
        self, x_train, y_train, x_val, y_val, model, num_classes, gamma: float = 0.5
    ) -> None:
        # local model
        self.model = instantiate(model)
        self.gamma = gamma
        # local dataset
        self.x_train, self.y_train = x_train, to_categorical(
            y_train, num_classes=num_classes
        )
        self.x_val, self.y_val = x_val, to_categorical(y_val, num_classes=num_classes)

    def get_parameters(self, config):
        """Return the parameters of the current local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        # Backup the current local model weights (wᵢ,ₜ₋₁)
        backup_weights = self.model.get_weights()

        # Create temporary model with old weights for gradient computation
        old_model = type(self.model)()  # Create new instance of same model type
        old_model.set_weights(backup_weights)

        # ----- Third Task: Update Local Model with the Sparsified Gradient -----
        # parameters here represent the sparsified gradient, vᵢ,ₜ.
        updated_weights = [w + delta for w, delta in zip(backup_weights, parameters)]
        # Set weights and train the model
        self.model.set_weights(updated_weights)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=config["local_epochs"],
            batch_size=config["batch_size"],
            verbose=False,
        )

        # ------ Secound
        # Compute normalized gradients
        normalized_gradients = self.compute_and_normalize_gradients(
            old_model=old_model,
            new_model=self.model,
            gamma=self.gamma,  # Using gamma as gamma scaling factor
        )
        # ------ First Task: Return the gradi
        return normalized_gradients, len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Implement distributed evaluation for a given client."""
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=False)
        return loss, len(self.x_val), {"accuracy": acc}

    def compute_and_normalize_gradients(self, old_model, new_model, gamma):
        """Compute and normalize gradients using L2 normalization.

        Args:
            old_model: Model before training
            new_model: Model after training
            gamma: Scaling factor (Γ)

        Returns
        -------
            Normalized gradients scaled by gamma
        """
        old_model_weights = old_model.get_weights()
        new_model_weights = new_model.get_weights()

        # Compute weight differences (∆wi,t)
        gradients = [
            new_model_weights[i] - old_model_weights[i]
            for i in range(len(new_model_weights))
        ]

        # Compute L2 norm of gradients (‖∆wi,t‖)
        # First flatten all gradients into a single array
        flattened_gradients = np.concatenate([g.flatten() for g in gradients])
        l2_norm = np.linalg.norm(flattened_gradients)

        # Normalize gradients and scale by gamma (Γ ∆wi,t /‖∆wi,t‖)
        if l2_norm > 0:  # Avoid division by zero
            normalized_gradients = [gamma * (grad / l2_norm) for grad in gradients]
        else:
            normalized_gradients = [np.zeros_like(grad) for grad in gradients]

        return normalized_gradients


def generate_client_fn(partitions, model, num_classes, gamma):
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        full_x_train_cid, full_y_train_cid = partitions[int(cid)]

        # Use 10% of the client's training data for validation
        split_idx = math.floor(len(full_x_train_cid) * 0.9)
        x_train_cid, y_train_cid = (
            full_x_train_cid[:split_idx],
            full_y_train_cid[:split_idx],
        )
        x_val_cid, y_val_cid = (
            full_x_train_cid[split_idx:],
            full_y_train_cid[split_idx:],
        )

        return FlowerClient(
            x_train_cid, y_train_cid, x_val_cid, y_val_cid, model, num_classes, gamma
        )

    return client_fn
