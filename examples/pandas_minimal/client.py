import warnings
import flwr as fl
import pandas as pd
import numpy as np


def load_data():
    df = pd.read_csv('./data/client.csv') 
    return df

df = load_data()

column_names = ['sepal length (cm)','sepal width (cm)']

def compute_hist(df, col_name):
    _, vals = np.histogram(df[col_name])  
    return vals

def compute_sum(df, col_name):
    val = df[col_name].sum()
    return val

def compute_min(df, col_name):
    val = df[col_name].min()
    return val

# Define Flower client
class FlowerClient(fl.client.Client):
    def get_parameters(self, config):
        return {'column_names':column_names}

    def set_parameters(self, parameters):
        pass
    
    def fit(self, parameters, config):
        # Execute query locally 
        outputs = {}
        v_arr = []
        for c in column_names:
            h = compute_hist(df, c)
            v_arr.append([c])
            v_arr.append(h) 
        return v_arr,len(df),{} # [Tensors, num_examples, dict] - I am storing analytics analysis in results

    def evaluate(self, parameters,data):
        pass
    
    def _format_outputs(self,inputs:dict):
        # Format outputs to [NDTupes, int, dict]
        metrics_dict = {}
        values_arr = []
        for index,key in enumerate(list(inputs.keys())):
            metrics_dict[str(index)] = key 
            values_arr.append(inputs[key])
        return values_arr, metrics_dict 

# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)