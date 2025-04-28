import torch
import torch.nn as nn
from config import cfg
from .utils import init_param, make_loss


class LSTM(nn.Module):
    def __init__(self, data_shape, hidden_size, num_layers, target_size):
        super().__init__()
        self.lstm = nn.LSTM(data_shape[1], hidden_size, num_layers, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, target_size)

    def f(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def forward(self, input):
        output = {}
        if 'data' in input:
            output['target'] = self.f(input['data'])
        if 'aug_data' in input:
            output['aug_target'] = self.f(input['aug_data'])
        if 'mix_data' in input:
            output['mix_target'] = self.f(input['mix_data'])
        output['loss'] = make_loss(output, input)
        return output


def lstm():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['lstm']['hidden_size']
    num_layers = cfg['lstm']['num_layers']
    model = LSTM(data_shape, hidden_size, num_layers, target_size)
    model.apply(init_param)
    return model
