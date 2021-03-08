"""
Flower client example using MXNet for MNIST classification.

The code is generally adapted from:

https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html
"""

import flwr as fl
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as F

# Fixing the random seed
mx.random.seed(42)

# Setup context to GPU or CPU
DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]


def main():
    def model():
        net = nn.Sequential()
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dense(10))
        net.collect_params().initialize()
        return net

    train_data, val_data = load_data()

    model = model()
    init = nd.random.uniform(shape=(2, 784))
    model(init)

    # Flower Client
    class MNISTClient(fl.client.NumPyClient):
        def get_parameters(self):
            param = []
            for val in model.collect_params(".*weight").values():
                p = val.data()
                param.append(p.asnumpy())
            return param

        def set_parameters(self, parameters):
            params = zip(model.collect_params(".*weight").keys(), parameters)
            for key, value in params:
                model.collect_params().setattr(key, value)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(model, train_data, epoch=1)
            return self.get_parameters(), train_data.batch_size, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(model, val_data)
            return float(loss),  val_data.batch_size, {"accuracy":float(accuracy)}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=MNISTClient())


def load_data():
    print("Download Dataset")
    mnist = mx.test_utils.get_mnist()
    batch_size = 100
    train_data = mx.io.NDArrayIter(
        mnist["train_data"], mnist["train_label"], batch_size, shuffle=True
    )
    val_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
    return train_data, val_data


def train(net, train_data, epoch):
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})
    metric = mx.metric.Accuracy()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for i in range(epoch):
        train_data.reset()
        for batch in train_data:
            data = gluon.utils.split_and_load(
                batch.data[0], ctx_list=DEVICE, batch_axis=0
            )
            label = gluon.utils.split_and_load(
                batch.label[0], ctx_list=DEVICE, batch_axis=0
            )
            outputs = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    loss = softmax_cross_entropy_loss(z, y)
                    loss.backward()
                    outputs.append(z)
            metric.update(label, outputs)
            trainer.step(batch.data[0].shape[0])
        name, acc = metric.get()
        metric.reset()
        print("training acc at epoch %d: %s=%f" % (i, name, acc))


def test(net, val_data):
    metric = mx.metric.Accuracy()
    loss_metric = mx.metric.Loss()
    loss = 0.0
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=DEVICE, batch_axis=0)
        label = gluon.utils.split_and_load(
            batch.label[0], ctx_list=DEVICE, batch_axis=0
        )
        outputs = []
        for x in data:
            outputs.append(net(x))
            loss_metric.update(label, outputs)
            loss += loss_metric.get()[1]
        metric.update(label, outputs)
    print("validation acc: %s=%f" % metric.get())
    print("validation loss:", loss)
    accuracy = metric.get()[1]
    return loss, accuracy


if __name__ == "__main__":
    main()
