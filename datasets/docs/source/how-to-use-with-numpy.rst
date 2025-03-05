Use with NumPy
==============

Let's integrate ``flwr-datasets`` with NumPy.

Create a ``FederatedDataset``::

  from flwr_datasets import FederatedDataset

  fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
  partition = fds.load_partition(0, "train")
  centralized_dataset = fds.load_split("test")

Inspect the names of the features::

  partition.features

In case of CIFAR10, you should see the following output.

.. code-block:: none

  {'img': Image(decode=True, id=None),
  'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
  'frog', 'horse', 'ship', 'truck'], id=None)}

We will use the keys in the partition features in order to apply transformations to the data or pass it to a ML model.  Let's move to the transformations.

NumPy
-----
Transform to NumPy::

  partition_np = partition.with_format("numpy")
  X_train, y_train = partition_np["img"], partition_np["label"]

That's all. Let's check the dimensions and data types of our ``X_train`` and ``y_train``::

  print(f"The shape of X_train is: {X_train.shape}, dtype: {X_train.dtype}.")
  print(f"The shape of y_train is: {y_train.shape}, dtype: {y_train.dtype}.")

You should see::

  The shape of X_train is: (500, 32, 32, 3), dtype: uint8.
  The shape of y_train is: (500,), dtype: int64.

Note that the ``X_train`` values are of type ``uint8``. It is not a problem for the TensorFlow model when passing the
data as input, but it might remind us to normalize the data - global normalization, pre-channel normalization, or simply
rescale the data to [0, 1] range::

  X_train = (X_train - X_train.mean()) / X_train.std() # Global normalization


CNN Keras model
---------------
Here's a quick example of how you can use that data with a simple CNN model::

  import tensorflow as tf
  from tensorflow.keras import datasets, layers, models

  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      layers.MaxPooling2D(2, 2),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D(2, 2),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=20, batch_size=64)

You should see about 98% accuracy on the training data at the end of the training.

Note that we used ``"sparse_categorical_crossentropy"``. Make sure to keep it that way if you don't want to one-hot-encode
the labels.
