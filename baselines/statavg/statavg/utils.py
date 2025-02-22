"""statavg: A Flower Baseline."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def evaluation_metrics(y_true, classes, predicted_test):
    """Not used in the current implementation.

    Auxiliary for generating additional results if needed.
    """
    # Accuracy
    accuracy = accuracy_score(y_true, classes)
    print(f"Accuracy: {accuracy}")

    cnf_matrix = confusion_matrix(y_true, classes)

    false_p = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    false_n = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    true_p = np.diag(cnf_matrix)
    true_n = cnf_matrix.sum() - (false_p + false_n + true_p)

    false_p = false_p.astype(float)
    false_n = false_n.astype(float)
    true_p = true_p.astype(float)
    true_n = true_n.astype(float)

    # true positive rate - TPR
    tpr = true_p / (true_p + false_n)
    print("TPR: ", np.mean(tpr))

    # false positive rate - FPR
    fpr = false_p / (false_p + true_n)
    print("FPR: ", np.mean(fpr))

    # F1 Score
    fsc = f1_score(y_true, classes, average="weighted")
    print(f"F1 score: {fsc}")

    auc = roc_auc_score(y_true, predicted_test, multi_class="ovr")
    print(f"AUC Score: {auc}")
    eval_metrics = (accuracy, fsc)

    return eval_metrics
