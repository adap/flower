"""Contains utility functions."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def evaluation_metrics(y_true, classes, predicted_test):
    """Not used in the current implementation.

    Auxiliary for generating additional results if needed.
    """
    # Accuracy
    accuracy = accuracy_score(y_true, classes)
    print("Accuracy: %f" % accuracy)

    cnf_matrix = confusion_matrix(y_true, classes)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # true positive rate - TPR
    TPR = TP / (TP + FN)
    print("TPR: ", np.mean(TPR))

    # false positive rate - FPR
    FPR = FP / (FP + TN)
    print("FPR: ", np.mean(FPR))

    # F1 Score
    f1 = f1_score(y_true, classes, average="weighted")
    print("F1 score: %f" % f1)

    auc = roc_auc_score(y_true, predicted_test, multi_class="ovr")
    print("AUC Score: %f" % auc)
    eval_metrics = (accuracy, f1)

    return eval_metrics


def plot_accuracy(results_path: str) -> None:
    """Plot the accuracy."""
    with open(results_path, "rb") as file:
        results = pickle.load(file)

    accuracy_dict = results["history"].metrics_distributed
    accuracy_lst = accuracy_dict["accuracy"]

    rounds = [p[0] for p in accuracy_lst]
    acc = [p[1] for p in accuracy_lst]

    plt.plot(rounds, acc, marker="o", linestyle="-")

    plt.xlabel("Rounds")
    plt.ylabel("Testing Accuracy")

    plt.grid(True)
    plt.show()
