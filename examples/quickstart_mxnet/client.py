"""Flower client example using MXNet for MNIST classification.

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
        net.add(nn.Dense(64, activation="relu"))
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
            [accuracy, loss], num_examples = train(model, train_data, epoch=2)
            results = {"accuracy": float(accuracy[1]), "loss": float(loss[1])}
            return self.get_parameters(), num_examples, results

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            [accuracy, loss], num_examples = test(model, val_data)
            print("Evaluation accuracy & loss", accuracy, loss)
            return float(loss[1]), num_examples, {"accuracy": float(accuracy[1])}

    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MNISTClient())


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
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.01})
    accuracy_metric = mx.metric.Accuracy()
    loss_metric = mx.metric.CrossEntropy()
    metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [accuracy_metric, loss_metric]:
        metrics.add(child_metric)
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for i in range(epoch):
        train_data.reset()
        num_examples = 0
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
                    outputs.append(z.softmax())
                    num_examples += len(x)
            metrics.update(label, outputs)
            trainer.step(batch.data[0].shape[0])
        trainings_metric = metrics.get_name_value()
        print("Accuracy & loss at epoch %d: %s" % (i, trainings_metric))
    return trainings_metric, num_examples


def test(net, val_data):
    accuracy_metric = mx.metric.Accuracy()
    loss_metric = mx.metric.CrossEntropy()
    metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [accuracy_metric, loss_metric]:
        metrics.add(child_metric)
    val_data.reset()
    num_examples = 0
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=DEVICE, batch_axis=0)
        label = gluon.utils.split_and_load(
            batch.label[0], ctx_list=DEVICE, batch_axis=0
        )
        outputs = []
        for x in data:
            outputs.append(net(x).softmax())
            num_examples += len(x)
        metrics.update(label, outputs)
    metrics.update(label, outputs)
    return metrics.get_name_value(), num_examples


if __name__ == "__main__":
    main()
