# %%
import argparse
from collections import OrderedDict
from typing import Dict, Tuple, List
from torch.utils.data import DataLoader
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

import flwr as fl


from utils_mnist import (
    test,
    visualize_gen_image,
    visualize_gmm_latent_representation,
    non_iid_train_iid_test,
    alignment_dataloader,
    train_align,
)
from utils_mnist import VAE
import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VAE()
trainsets, valsets = non_iid_train_iid_test()
testset = valsets[-1]
model.to(device)

testloader = DataLoader(testset, batch_size=64)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt




def latent_stats(model, test_loader, device):
    model.eval()
    all_latents = []
    all_labels = []
    all_means = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            z, mu, _ = model(data)
            all_latents.append(z.cpu().numpy())
            all_means.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_means = np.concatenate(all_means, axis=0)
    return all_latents, all_means, all_labels


def model_weights(weights_path):
    with open(weights_path, "rb") as f:
        weights = np.load(f, allow_pickle=True)
    return weights.tolist()


set_params(model, model_weights(weights_path))
all_latents, all_means, all_labels = latent_stats(model, testloader, device)
# # %%
# set_params(model, parameters_prox_0.tolist())
# visualize_gmm_latent_representation(model, testloader, device)

# # %%
# set_params(model, parameters_cir_0.tolist())
# visualize_gmm_latent_representation(model, testloader, device)

# # %%
# set_params(model, parameters_prox_10.tolist())
# visualize_gmm_latent_representation(model, testloader, device)

# # %%
# set_params(model, parameters_cir_10.tolist())
# visualize_gmm_latent_representation(model, testloader, device)

# # %%
# set_params(model, parameters_cir_reg_10.tolist())
# visualize_gmm_latent_representation(model, testloader, device)

# # %%
# set_params(model, parameters_avg_10.tolist())
# visualize_gmm_latent_representation(model, testloader, device)

# # %%
# set_params(model, parameters_avg_10.tolist())
# all_latents_avg_10, all_means_avg_10, all_labels_avg_10 = latent_stats(
#     model, testloader, device
# )
# reid_evaluation(all_means_avg_10, all_labels_avg_10)

# # %%
# set_params(model, parameters_cir_reg_10.tolist())
# all_latents_cir_reg_10, all_means_cir_reg_10, all_labels_cir_reg_10 = latent_stats(
#     model, testloader, device
# )
# reid_evaluation(all_means_cir_reg_10, all_labels_cir_reg_10)

# # %%
# set_params(model, parameters_cir_10.tolist())
# all_latents_cir_10, all_means_cir_10, all_labels_cir_10 = latent_stats(
#     model, testloader, device
# )

from scipy.spatial import distance_matrix
from pprint import pprint


def reid_evaluation(embeddings, labels):
    mat = distance_matrix(embeddings, embeddings)

    same_distances = []
    diff_distances = []

    dev_ind = [np.where(np.array(labels) == i)[0] for i in range(10)]
    for i in range(10):
        for j in range(i, 10):
            sub_mat = mat[dev_ind[i]].T
            subsub_mat = sub_mat[dev_ind[j]]
            for ind_i in range(len(subsub_mat)):
                for ind_j in range(ind_i + 1, len(subsub_mat[0])):
                    d = subsub_mat[ind_i][ind_j]
                    if i == j:
                        same_distances.append(d)
                    else:
                        diff_distances.append(d)

    roc_value = []
    th_steps = 50
    max_dist = max(diff_distances)
    min_dist = min(same_distances)

    for th in range(th_steps + 1):
        t = (th + 0.5) / th_steps
        t = t * t
        threshold = t * max_dist + (1 - t) * min_dist
        #         threshold = threshold * threshold
        ta = len(np.where(np.array(same_distances) < threshold)[0]) / len(
            same_distances
        )
        fa = len(np.where(np.array(diff_distances) < threshold)[0]) / len(
            diff_distances
        )
        roc_value.append([fa, ta, threshold])
    roc_value = np.array(roc_value)
    area = np.trapz(roc_value[:, 1], roc_value[:, 0])

    # Get accuracy when fa=0.1% , 1%, 10%
    targets_fa = [0.001, 0.01, 0.1]
    current_target_fa_id = 0
    target_th = []
    ta_accuracy = []
    for i in range(len(roc_value)):
        if roc_value[i][0] >= targets_fa[current_target_fa_id]:
            target_th.append(roc_value[i][2])
            ta_accuracy.append(roc_value[i][1])
            current_target_fa_id += 1
            if current_target_fa_id >= len(targets_fa):
                break

    # Nearest Neighbor accuracy
    correct_classification = 0
    wrong_classification = 0
    nearest10_amt = 0
    for i in range(len(mat)):
        nearest_id = np.argpartition(mat[i], 2)[1]
        nearest_10id = np.argpartition(mat[i], 11)[1:11]
        if labels[nearest_id] == labels[i]:
            correct_classification += 1
        else:
            wrong_classification += 1

        nearest10_amt += len(np.where(np.array(labels)[nearest_10id] == labels[i])[0])

    plt.plot(roc_value[:, 0], label="Positive")
    plt.plot(roc_value[:, 1], label="Negative")

    # Add title and legend
    plt.title("Clustering Metric")
    plt.legend(["Positive", "Negative"])

    # Show the plot
    pprint(
        {
            "Positive clustering at error rate 0.1%": 100 * ta_accuracy[0],
            "Positive clustering at error rate 1%": 100 * ta_accuracy[1],
            "Positive clustering at error rate 10%": 100 * ta_accuracy[2],
            "Trapz": area,
            "NN classification on test data": 100
            * correct_classification
            / (correct_classification + wrong_classification),
        }
    )
    plt.show()






# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    all_means_cir_10, all_labels_cir_10, test_size=0.8, random_state=42
)

# Define XGBoost parameters for multi-class classification
params = {
    "objective": "multi:softprob",  # Use softmax for multi-class problems
    "eval_metric": "mlogloss",  # Multiclass logloss metric
    "num_class": 10,  # Number of classes (change to 10 for your dataset)
    "use_label_encoder": False,  # To avoid a warning about deprecation
}

# Create DMatrix for training
dtrain = xgb.DMatrix(X_train, label=y_train)

# Train the XGBoost classifier
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)

# Create DMatrix for testing
dtest = xgb.DMatrix(X_test)

# Make predictions
y_pred_probs = xgb_model.predict(dtest)

# Convert predicted probabilities to class predictions
y_pred = y_pred_probs.argmax(axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    all_means_prox_10, all_labels_prox_10, test_size=0.8, random_state=42
)

# Define XGBoost parameters for multi-class classification
params = {
    "objective": "multi:softprob",  # Use softmax for multi-class problems
    "eval_metric": "mlogloss",  # Multiclass logloss metric
    "num_class": 10,  # Number of classes (change to 10 for your dataset)
    "use_label_encoder": False,  # To avoid a warning about deprecation
}

# Create DMatrix for training
dtrain = xgb.DMatrix(X_train, label=y_train)

# Train the XGBoost classifier
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)

# Create DMatrix for testing
dtest = xgb.DMatrix(X_test)

# Make predictions
y_pred_probs = xgb_model.predict(dtest)

# Convert predicted probabilities to class predictions
y_pred = y_pred_probs.argmax(axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

set_params(model, parameters_prox_0.tolist())
all_latents_prox_0, all_means_prox_0, all_labels_prox_0 = latent_stats(
    model, testloader, device
)



# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    all_means_prox_0, all_labels_prox_0, test_size=0.8, random_state=42
)

# Define XGBoost parameters for multi-class classification
params = {
    "objective": "multi:softprob",  # Use softmax for multi-class problems
    "eval_metric": "mlogloss",  # Multiclass logloss metric
    "num_class": 10,  # Number of classes (change to 10 for your dataset)
    "use_label_encoder": False,  # To avoid a warning about deprecation
}

# Create DMatrix for training
dtrain = xgb.DMatrix(X_train, label=y_train)

# Train the XGBoost classifier
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)

# Create DMatrix for testing
dtest = xgb.DMatrix(X_test)

# Make predictions
y_pred_probs = xgb_model.predict(dtest)

# Convert predicted probabilities to class predictions
y_pred = y_pred_probs.argmax(axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

set_params(model, parameters_avg_10.tolist())
all_latents_avg_10, all_means_avg_10, all_labels_avg_10 = latent_stats(
    model, testloader, device
)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    all_means_avg_10, all_labels_avg_10, test_size=0.8, random_state=42
)

# Define XGBoost parameters for multi-class classification
params = {
    "objective": "multi:softprob",  # Use softmax for multi-class problems
    "eval_metric": "mlogloss",  # Multiclass logloss metric
    "num_class": 10,  # Number of classes (change to 10 for your dataset)
    "use_label_encoder": False,  # To avoid a warning about deprecation
}

# Create DMatrix for training
dtrain = xgb.DMatrix(X_train, label=y_train)

# Train the XGBoost classifier
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)

# Create DMatrix for testing
dtest = xgb.DMatrix(X_test)

# Make predictions
y_pred_probs = xgb_model.predict(dtest)

# Convert predicted probabilities to class predictions
y_pred = y_pred_probs.argmax(axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

reid_evaluation(all_means_avg_10, all_labels_avg_10)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    all_means_cir_reg_10, all_labels_cir_reg_10, test_size=0.8, random_state=42
)

# Define XGBoost parameters for multi-class classification
params = {
    "objective": "multi:softprob",  # Use softmax for multi-class problems
    "eval_metric": "mlogloss",  # Multiclass logloss metric
    "num_class": 10,  # Number of classes (change to 10 for your dataset)
    "use_label_encoder": False,  # To avoid a warning about deprecation
}

# Create DMatrix for training
dtrain = xgb.DMatrix(X_train, label=y_train)

# Train the XGBoost classifier
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)

# Create DMatrix for testing
dtest = xgb.DMatrix(X_test)

# Make predictions
y_pred_probs = xgb_model.predict(dtest)

# Convert predicted probabilities to class predictions
y_pred = y_pred_probs.argmax(axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

