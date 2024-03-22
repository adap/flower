# Example using PyTorch

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

To start with, install the dependencies listed here:
https://github.com/adap/flower/examples/quickstart-pytorch

Then, we need to install my fork of Flower with the new communication protocol:
https://github.com/brianlck/flower

Finally, if you are using S3 as an external storage, please configure boto3 as documented here:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html