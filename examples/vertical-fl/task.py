import torch


def load_data():
    # Create some dummy data
    X = torch.randn(100, 5)  # 100 samples, 5 features
    y = (torch.sum(X, dim=1) > 0).float()  # Sum of features > 0 as positive label

    # Split the data
    X_train = X[:80]
    y_train = y[:80]
    X_test = X[80:]
    y_test = y[80:]
    return (X_train, y_train), (X_test, y_test)
