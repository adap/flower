import json
import numpy as np

from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, data_root):
        self.CHARACTERS = '''1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -,\'!;"[]?().:'''
        self.NUM_LETTERS = len(self.ALL_LETTERS) # 76

        with open(data_root) as file:
            js=json.load(file)
            self.x, self.y = [], []
            for u in js['users']:
                self.x+=js['user_data'][u]['x']
                self.y+=js['user_data'][u]['y']

    def word_to_indices(self, word):
        indices = [ self.CHARACTERS.find(c) for c in word ]
        return indices
	
    def __len__(self):
	    return len(self.y)
	
    def __getitem__(self, idx):
    	x = np.array(self.word_to_indices(self.x[idx]))
        y = np.array(self.CHARACTERS.find(self.y[idx]))
        return x, y 