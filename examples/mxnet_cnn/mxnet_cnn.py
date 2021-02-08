"""MXNet MNIST image classification.

The code is generally adapted from:

https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html#Loading-Data
"""

from __future__ import print_function
from typing import Tuple
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as F

# Fixing the random seed
mx.random.seed(42)


def load_data() -> Tuple[mx.io.NDArrayIter, mx.io.NDArrayIter]:
    print("Download Dataset")
    # Download MNIST data
    mnist = mx.test_utils.get_mnist()
    batch_size = 100
    train_data = mx.io.NDArrayIter(
        mnist["train_data"], mnist["train_label"], batch_size, shuffle=True
    )
    val_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
    return train_data, val_data


print("Define CNN")


class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(20, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv2 = nn.Conv2D(50, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.fc1 = nn.Dense(500)
        self.fc2 = nn.Dense(10)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x


def train(
    net: Net, train_data: mx.io.NDArrayIter, epoch: int, device: mx.context
) -> None:
    # print('Xavier Init', mx.init.Xavier(magnitude=2.24))

    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})
    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    loss_metric = mx.metric.Loss()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    print("Start MXNet training")
    for i in range(epoch):
        # Reset the train data iterator.
        train_data.reset()
        # Loop over the train data iterator.
        for batch in train_data:
            # Splits train data into multiple slices along batch_axis
            # and copy each slice into a context.
            data = gluon.utils.split_and_load(
                batch.data[0], ctx_list=device, batch_axis=0
            )
            # Splits train labels into multiple slices along batch_axis
            # and copy each slice into a context.
            label = gluon.utils.split_and_load(
                batch.label[0], ctx_list=device, batch_axis=0
            )
            outputs = []
            # Inside training scope
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    # Computes softmax cross entropy loss.
                    loss = softmax_cross_entropy_loss(z, y)
                    # Backpropogate the error for one iteration.
                    loss.backward()
                    outputs.append(z)
            # Updates internal evaluation
            metric.update(label, outputs)
            # loss_metric.update(label, loss)
            # Make one step of parameter update. Trainer needs to know the
            # batch size of data to normalize the gradient by 1/batch_size.
            trainer.step(batch.data[0].shape[0])
        # Gets the evaluation result.
        name, acc = metric.get()
        # name_loss, running_loss = loss_metric.get()
        # Reset evaluation result to initial state.
        metric.reset()
        print("training acc at epoch %d: %s=%f" % (i, name, acc))
        # print('training loss at epoch %d: %s=%f'%(i, name_loss, running_loss))


def test(
    net: Net, val_data: mx.io.NDArrayIter, device: mx.context
) -> Tuple[float, float]:
    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    loss_metric = mx.metric.Loss()
    loss = 0.0
    eval_loss = 0.0
    print("start batch processing")
    # Reset the validation data iterator.
    val_data.reset()
    # Loop over the validation data iterator.
    for batch in val_data:
        # Splits validation data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=device, batch_axis=0)
        # Splits validation label into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(
            batch.label[0], ctx_list=device, batch_axis=0
        )
        outputs = []
        for x in data:
            outputs.append(net(x))
            loss_metric.update(label, outputs)
            loss += loss_metric.get()[1]
        # Updates internal evaluation
        metric.update(label, outputs)
    print("validation acc: %s=%f" % metric.get())
    accuracy = metric.get()[1]
    # assert metric.get()[1] > 0.98
    return loss, accuracy


def main():
    print("Setup context to GPU and if not available to CPU")
    DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
    train_data, val_data = load_data()
    print("train_data", train_data, len(list(train_data)))
    net = Net()
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=DEVICE)
    train(net=net, train_data=train_data, epoch=2, device=DEVICE)
    loss, acc = test(net=net, val_data=val_data, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", acc)
    print("print key", net.collect_params().keys())
    print("Collect Parameters", net.collect_params().values())
    # for val in  net.collect_params():
    #    print('Parameter value',val)
    # parameters=[val.data() for val in net.collect_params().values()]


if __name__ == "__main__":
    main()
