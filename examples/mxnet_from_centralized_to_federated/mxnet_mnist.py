"""MXNet MNIST image classification.

The code is generally adapted from:

https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html
"""


from typing import List, Tuple
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as F
from mxnet import nd

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


def model():
    # Define simple Sequential model
    net = nn.Sequential()
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(64, activation="relu"))
    net.add(nn.Dense(10))
    net.collect_params().initialize()
    return net


def train(
    net: mx.gluon.nn, train_data: mx.io.NDArrayIter, epoch: int, device: mx.context
) -> Tuple[List[float], int]:
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.01})
    # Use Accuracy and Cross Entropy Loss as the evaluation metric.
    accuracy_metric = mx.metric.Accuracy()
    loss_metric = mx.metric.CrossEntropy()
    metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [accuracy_metric, loss_metric]:
        metrics.add(child_metric)
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for i in range(epoch):
        # Reset the train data iterator.
        train_data.reset()
        # Calculate number of samples
        num_examples = 0
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
                    outputs.append(z.softmax())
                    num_examples += len(x)
            # Updates internal evaluation
            metrics.update(label, outputs)
            # Make one step of parameter update. Trainer needs to know the
            # batch size of data to normalize the gradient by 1/batch_size.
            trainer.step(batch.data[0].shape[0])
        # Gets the evaluation result.
        trainings_metric = metrics.get_name_value()
        print("Accuracy & loss at epoch %d: %s" % (i, trainings_metric))
    return trainings_metric, num_examples


def test(
    net: mx.gluon.nn, val_data: mx.io.NDArrayIter, device: mx.context
) -> Tuple[List[float], int]:
    # Use Accuracy as the evaluation metric.
    accuracy_metric = mx.metric.Accuracy()
    loss_metric = mx.metric.CrossEntropy()
    metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [accuracy_metric, loss_metric]:
        metrics.add(child_metric)
    # Reset the validation data iterator.
    val_data.reset()
    # Get number of samples for val_dat
    num_examples = 0
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
            outputs.append(net(x).softmax())
            num_examples += len(x)
        # Updates internal evaluation
        metrics.update(label, outputs)
    return metrics.get_name_value(), num_examples


def main():
    # Set context to GPU or - if not available - to CPU
    DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
    # Load train and validation data
    train_data, val_data = load_data()
    # Define sequential model
    net = model()
    init = nd.random.uniform(shape=(2, 784))
    net(init)
    # Start model training based on training set
    train(net=net, train_data=train_data, epoch=2, device=DEVICE)
    # Evaluate model using loss and accuracy
    eval_metric, _ = test(net=net, val_data=val_data, device=DEVICE)
    acc = eval_metric[0]
    loss = eval_metric[1]
    print("Evaluation Loss: ", loss)
    print("Evaluation Accuracy: ", acc)


if __name__ == "__main__":
    main()
