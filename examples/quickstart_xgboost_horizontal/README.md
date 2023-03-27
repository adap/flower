# Federated XGBoost in Horizontal Setting (PyTorch)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/examples/quickstart_xgboost_horizontal/code_horizontal.ipynb) (or open the [Jupyter Notebook](https://github.com/adap/flower/blob/main/examples/quickstart_xgboost_horizontal/code_horizontal.ipynb))

This example demonstrates a federated XGBoost using Flower with PyTorch. This is a novel method to conduct federated XGBoost in the horizontal setting. It differs from the previous methods in the following ways:

- We aggregate and conduct federated learning on client tree’s prediction outcomes by sending clients' built XGBoost trees to the server and then sharing to the clients.
- The exchange of privacy-sensitive information (gradients) is not needed.
- The model is a CNN with 1D convolution kernel size = the number of XGBoost trees in the client tree ensembles. 
- Using 1D convolution, we make the tree learning rate (a hyperparameter of XGBoost) learnable.

## Project Setup

This implementation can be easily run in Google Colab with the button at the top of the README or as a standalone Jupyter notebook,
it will automatically download and extract the example data inside a `dataset` folder and `binary_classification` and `regression` sub-folders.

## Datasets

This implementation supports both binary classification and regression datasets in SVM light format, loaded from ([LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)). Simply download the dataset files from the website and put them in the folder location indicated above.
