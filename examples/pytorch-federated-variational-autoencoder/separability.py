# %%
import torch
from utils_mnist import VAE, non_iid_train_iid_test, set_params
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 16
import numpy as np

test_loader = DataLoader(non_iid_train_iid_test()[1][-1], batch_size=128, shuffle=True)
model = VAE(z_dim=LATENT_DIM).to(DEVICE)
# Encode your data points into the latent space


def model_weights(weights_path):
    with open(weights_path, "rb") as f:
        weights = np.load(f, allow_pickle=True)
    return weights.tolist()


set_params(
    model,
    model_weights("cir_cls10_kl_gen_dim16/earthy-sweep-1-weights_cir_round_0.npy"),
)

# %%
model.eval()

# %%
latent_points = []
labels_list = []
for inputs, labels in test_loader:
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    out, mu, _ = model(inputs)
    latent_points.append(mu)
    labels_list.append(labels)
latent_points = torch.cat(latent_points, dim=0)
labels = torch.cat(labels_list, dim=0)
latent_points = latent_points.to(DEVICE)
labels = labels.to(DEVICE)

# %%


# Compute the mean of each class in the latent space
latent_means = {}
for class_label in torch.unique(labels):
    indices = torch.where(labels == class_label)[0]
    points = latent_points[indices]
    mean = torch.mean(points, dim=0)
    latent_means[class_label.item()] = mean

# Separate the encoded points based on their corresponding classes
latent_points_per_class = {}
for class_label in torch.unique(labels):
    indices = torch.where(labels == class_label)[0]
    points = latent_points[indices]
    latent_points_per_class[class_label.item()] = points

# Compute the within-class scatter matrix
within_class_scatter = torch.zeros(LATENT_DIM, LATENT_DIM, device=DEVICE)
for class_label, points in latent_points_per_class.items():
    mean = latent_means[class_label]
    centered_points = points - mean
    scatter_matrix = torch.matmul(centered_points.T, centered_points)
    within_class_scatter += scatter_matrix

# Compute the between-class scatter matrix
overall_mean = torch.mean(latent_points, dim=0)
between_class_scatter = torch.zeros(LATENT_DIM, LATENT_DIM,device=DEVICE)
for class_label, mean in latent_means.items():
    mean_diff = mean - overall_mean
    between_class_scatter += len(latent_points_per_class[class_label]) * torch.outer(
        mean_diff, mean_diff
    )

# Compute Fisher's linear discriminant ratio
Fisher_ratio = torch.trace(
    torch.matmul(torch.inverse(within_class_scatter), between_class_scatter)
)

print("Fisher's linear discriminant ratio:", Fisher_ratio.item())

# %%
