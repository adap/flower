# Federated XGBoost in Horizontal Setting (PyTorch)

This example demonstrates a federated XGBoost using Flower with PyTorch. This is a novel method to conduct federated XGBoost in the horizontal setting. It differs from the previous methods in the following ways:

- We aggregate and conduct federated learning on client tree’s prediction outcomes by sending clients' built XGBoost trees to the server and then sharing to the clients.
- The exchange of privacy-sensitive information (gradients) is not needed.
- The model is a CNN with 1D convolution kernel size = the number of XGBoost trees in the client tree ensembles. 
- Using 1D convolution, we make the tree learning rate (a hyperparameter of XGBoost) learnable.

## Project Setup

This implementation can be easily run in Google Colab with the following file structure in Google Drive, * denotes folder:

```shell
—————————————————————————————————————————————————————————————————————
My Drive
  XGBoost*
      |----- code.ipynb
      dataset*
          binary_classifications*
              |----- dataset file 1
              |----- dataset file 2
          regression*
              |----- dataset file 1
              |----- dataset file 2 
—————————————————————————————————————————————————————————————————————
```

## Datasets

This implementation supports both binary classification and regression datasets in SVM light format, loaded from ([LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)). Simply download the dataset files from the website and put them in the folder location indicated above.
