import torch
import torch.nn as nn

from flwr.eaaselines.datasets.leaf.models.utils.language_utils import ALL_LETTERS


class ShakespeareLeafNet(nn.Module):
    def __init__(self, chars:str=ALL_CHARACTERS, seq_len:int = 80,
                 hidden_size:int = 256, embedding_dim = 8):
        super().__init__()
        self.dict_size = len(chars)
        self.seq_len = seq_len 
        self.hidden_size= hidden_size

        self.encoder = nn.Embedding(self.dict_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_size,
                            num_layers = 2)
        self.decoder = nn.Linear(hidden_size*seq_len, self.dict_size)
        
    def forward(self, x):
        encoded_seq = self.encoder(x)
        outputs, (h_n, c_n) = self.lstm(encoded_seq) # (seq_len, batch,  hidden_size)
        pred = self.decoder( outputs[-1,:,:].view(-1, self.hidden_size*seq_len) )
        return pred
    

def get_model():
    return ShakespeareLeafNet()

if __name__ == '__main__':
    a = get_model()
    b =  torch.zeros( (90, 1, len(ALL_CHARACTERS)), dtype=torch.int64)
    print(b.shape)
    a(b)