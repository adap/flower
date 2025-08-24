from utils import sim_data
import pickle
import numpy as np
import bz2

dataset = sim_data(200, 25, 1000, .1, .1)
# datase2 = sim_data(500, 25, 1000, .1, .1)
# X1, y1 = sim_data(100, 1, 1000, .1, .1)
# X2, y2 = sim_data(100, 1, 1000, .1, .1)
# X3, y3 = sim_data(100, 1, 1000, .1, .1)
# X4, y4 = sim_data(100, 1, 1000, .1, .1)
# X5, y5 = sim_data(100, 1, 1000, .1, .1)
# X6, y6 = sim_data(100, 1, 1000, .1, .1)
# X7, y7 = sim_data(100, 1, 1000, .1, .1)
# X8, y8 = sim_data(100, 1, 1000, .1, .1)
# X9, y9 = sim_data(100, 1, 1000, .1, .1)
# X10, y10 = sim_data(100, 1, 1000, .1, .1)

# X = np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10), axis = 0)
# y = np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10), axis = 0)
# test_dataset = X, y

X, y, Xtest, ytest = dataset
train_dataset = X, y
test_dataset = Xtest, ytest

# print(Xtest)
# print(ytest)
filename_train = 'simII_train.bz2'
filename_test = 'simII_test.bz2'

with bz2.open(filename_train, "wb") as file:
    pickle.dump(train_dataset, file)

with bz2.open(filename_test, "wb") as file:
     pickle.dump(test_dataset, file)

# with gzip.open('fedht/data/simII_train.pkl', 'rb') as file:
#     dataset = pickle.load(file)

# with gzip.open('fedht/data/simII_test.pkl', 'rb') as file:
#     test_dataset = pickle.load(file)

# X_test, y_test = test_dataset
# X_test, y_test = test_dataset

# print(X_test)