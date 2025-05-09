"""Prepare the CICIDS2018 dataset for federated learning."""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Load the raw CSV dataset
# ----------------------------
def load_data(path_to_dataset: str):
    """
    Loads and cleans the dataset from the given directory.

    - Replaces infinities with NaN
    - Drops rows with NaN values
    - Returns a cleaned pandas DataFrame
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, path_to_dataset, "dataset.csv")
    dataset = pd.read_csv(dataset_path)

    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)

    dataset.info()
    return dataset

# ----------------------------
# Select specific features + encode labels
# ----------------------------
def pre_process_dataset(dataset: pd.DataFrame):
    """
    Select relevant features and convert string labels to numeric class indices.
    """
    selected_columns = [
        'Dst Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Bwd Pkt Len Mean',
        'Bwd Pkt Len Std', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean',
        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Mean', 'Bwd IAT Std',
        'Bwd IAT Max', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
        'Bwd Pkts/s', 'Pkt Len Max', 'Pkt Len Var', 'Bwd Seg Size Avg',
        'Subflow Fwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
        'Fwd Seg Size Min', 'Label'
    ]

    data = dataset.loc[:, selected_columns]
    data.dropna()

    class2idx = {
        "Bot": 0, "Benign": 1, "Infilteration": 2, "DoS attacks-GoldenEye": 3,
        "DoS attacks-Hulk": 4, "DoS attacks-SlowHTTPTest": 5, "DDOS attack-HOIC": 6,
        "DoS attacks-Slowloris": 7, "SSH-Bruteforce": 8, "DDOS attack-LOIC-UDP": 9,
        "Brute Force -Web": 10, "FTP-BruteForce": 11, "Brute Force -XSS": 12,
        "SQL Injection": 13, "DDoS attacks-LOIC-HTTP": 14
    }

    # Replace labels with integers
    data['Label'].replace(class2idx, inplace=True)
    data['Label'] = data['Label'].astype('float')

    return data

# ----------------------------
# Filter out labels manually or randomly
# ----------------------------
def filter_training_labels(train_df: pd.DataFrame, drop_labels: list = None,
                           random_drop: bool = True, min_random_drop: int = 1, num_drop: int=1):
    """
    Filters out specific or randomly selected labels from the training data.
    """
    if 'Label' not in train_df.columns:
        raise ValueError("DataFrame must contain a 'Label' column.")
    
    if random_drop:
        all_labels = train_df['Label'].unique().tolist()
        num = random.randint(min_random_drop, num_drop)
        drop_labels = random.sample(all_labels, min(num, len(all_labels) - 1))
    elif drop_labels is None:
        drop_labels = []
    print(num)
    print(drop_labels)
    filtered_df = train_df[~train_df['Label'].isin(drop_labels)].reset_index(drop=True)
    num_classes = filtered_df['Label'].nunique()
    print("existing classes:", filtered_df['Label'].unique())
    print("number of remaining classes:", num_classes)
    return filtered_df, drop_labels, num_classes

# ----------------------------
# Normalize features & create PyTorch Datasets
# ----------------------------
def scale_datasets(train, test, y_col='Label'):
    """
    Scales feature values using MinMaxScaler and wraps them into PyTorch Datasets.
    """
    X_train, y_train = train.drop(columns=[y_col]), train[y_col]
    X_val, y_val = test.drop(columns=[y_col]), test[y_col]

    print("Training set label distribution:")
    print(train[y_col].value_counts())
    num_features = len(X_train.columns)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    class ClassifierDataset(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    return train_dataset, val_dataset, num_features

# ----------------------------
# Main data preparation function
# ----------------------------
def prepare_dataset(path_to_dataset: str, num_clients: int, batch_size: int, val_ratio: float,  num_drop:int, total_classes:int):
    """
    Loads, processes, splits, and partitions CICIDS2018 dataset into federated clients.

    Returns:
        trainloaders (List[DataLoader]): Training data per client.
        valloaders (List[DataLoader]): Validation data per client.
    """
    dataset = load_data(path_to_dataset)
    dataset = pre_process_dataset(dataset)

    rows_per_client = len(dataset) // num_clients
    remaining = dataset.copy()
    clientsets = []

    # Partition the dataset equally among clients
    for _ in range(num_clients - 1):
        client_df, remaining = train_test_split(
            remaining,
            train_size=rows_per_client,
            random_state=2023,
            stratify=remaining ['Label']
        )
        clientsets.append(client_df.reset_index(drop=True))

    # Last client gets the remaining rows (including leftovers)
    clientsets.append(remaining.reset_index(drop=True))

    trainloaders = []
    valloaders = []
    num_classes=[]
    # Prepare DataLoaders for each client
    for trainset_ in clientsets:
        client_train, client_test = train_test_split(
            trainset_,
            test_size=val_ratio,
            random_state=2023,
            stratify=trainset_['Label']
        )
        if trainset_ is clientsets[-1]:
        # Special behavior for last
          #print("Last clientset:")
         # print("existing classes:", client_train['Label'].unique())
          num_classes_cl= total_classes
        else:
          #print("Regular clientset:")
        
        # Optionally drop random labels from client training data
          client_train, dropped, num_classes_cl = filter_training_labels(
            client_train,
            random_drop=True,
            
            num_drop=num_drop
          )

        # Scale and convert to PyTorch Datasets
        train_dataset, val_dataset, num_features = scale_datasets(client_train, client_test)

        # Create PyTorch DataLoaders
        trainloaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0))
        valloaders.append(DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0))
        num_classes.append(num_classes_cl)
    return trainloaders, valloaders, num_classes, num_features
