from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

def partition_dataset(x, y, num_clients, client_id):
    data_size = len(x)
    partition_size = data_size // num_clients

    start = partition_size * client_id
    end = start + partition_size if client_id != num_clients - 1 else data_size

    x_client = x[start:end]
    y_client = y[start:end]
    return x_client, y_client

def load_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_metrics(loss_per_round, accuracy_per_round):
    """Plot the loss and accuracy at the end of execution."""
    rounds = list(range(1, len(loss_per_round) + 1))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.plot(rounds, loss_per_round, label="Loss", marker="o")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss per Round")
    plt.grid(True)

    plt.subplot(1, 2, 1)
    plt.plot(rounds, accuracy_per_round, label="Accuracy", marker="o", color="green")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Round")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("metrics.jpg", format="jpg")
    plt.close()