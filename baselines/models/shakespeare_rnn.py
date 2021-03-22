import torch

class ShakespeareLeafNet(nn.Module):
    def __init__(self, characters:str, seq_len:int = 80, hidden_size:int = 256,
                 embedding_size = 8, output_size=80):
        super().__init__()
        self.dict_size = len(characters)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding_dim = embedding_dim
        self.encoder = nn.Embedding(self.dict_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_size,
                            num_layers = 2,
                            batch_first = True)
        self.decoder=nn.Linear(hidden_size*input_size, output_size)
        
    def forward(self, x):
        x=self.encoder(x)
        zx, hidden = self.lstm(x)
        output=self.decoder(zx.contiguous().view(zx.shape[0], -1))
        return output 
    

def get_model():
    return ShakespeareLeafNet()