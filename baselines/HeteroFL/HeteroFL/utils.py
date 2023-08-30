"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import torch
import numpy as np

from models import Conv



def preprocess_input(cfg_model , cfg_data):
    model_config = {}
    if(cfg_model.model_name == "conv"):
        model_config['model_name'] = "Conv"
    # elif for others...

    if(cfg_data.dataset_name == 'MNIST'):
        model_config['data_shape'] = (1 , 28 , 28)
        model_config['classes_size'] = 10
    
    model_config['hidden_layers'] = cfg_model.hidden_layers

    return model_config

def get_global_model_rate(model_mode):
    model_mode = '' + model_mode
    model_mode = model_mode.split('-')[0][0]
    return model_mode


class Model_rate_manager():
    def __init__(self , model_split_mode , model_split_rate , model_mode ):
        self.model_split_mode = model_split_mode
        self.model_split_rate = model_split_rate
        self.model_mode = model_mode

    def create_model_rate_mapping(self , num_users):
        client_model_rate = []
        self.model_mode = self.model_mode.split('-')

        if self.model_split_mode == 'fix':
            mode_rate , proportion = [] , []
            for m in self.model_mode:
                mode_rate.append(self.model_split_rate[m[0]])
                proportion.append(int(m[1:]))
            print("King of Kothaaaaa" , len(mode_rate))
            num_users_proportion = num_users // sum(proportion)
            print('num_of_users_proportion = ' , num_users_proportion)
            print('num_users = ' , num_users , 'sum(prportion = )' , sum(proportion))
            for i in range(len(mode_rate)):
                client_model_rate += np.repeat(mode_rate[i], num_users_proportion * proportion[i]).tolist()

            print('that minus = ' , num_users - len(client_model_rate) , 'len of client model_rate = ' , len(client_model_rate))
            for i in range(num_users - len(client_model_rate)):
                print(client_model_rate[-1])
            client_model_rate = client_model_rate + [client_model_rate[-1] for _ in
                                                     range(num_users - len(client_model_rate))]
            return client_model_rate

        elif (self.model_split_mode == 'dynamic'):
            mode_rate , proportion = [] , []

            for m in self.model_mode:
                mode_rate.append(self.model_split_rate[m[0]])
                proportion.append(int(m[1:]))

            proportion = (np.array(proportion) / sum(proportion)).tolist()

            rate_idx = torch.multinomial(torch.tensor(proportion), num_samples=num_users,
                                             replacement=True).tolist()
            client_model_rate = np.array(mode_rate)[rate_idx]

            return client_model_rate

        else :
            return None