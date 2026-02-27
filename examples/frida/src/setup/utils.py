import numpy as np


def generate_random_indices(num_training_rounds, num_data_samples, len_loader, batch_size):
    random_indices = []
    for _ in range(num_training_rounds):
        random_indices_batches = []
        for _ in range(len_loader):
            indices = list(np.random.randint(0, batch_size, size=num_data_samples))
            random_indices_batches.append(indices)
        random_indices.append(random_indices_batches)

    return random_indices
