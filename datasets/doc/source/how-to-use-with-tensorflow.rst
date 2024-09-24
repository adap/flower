Use with TensorFlow
===================

Let's integrate ``flwr-datasets`` with ``TensorFlow``. We show you three ways how to convert the data into the formats
that ``TensorFlow``'s models expect.  Please note that, especially for the smaller datasets, the performance of the
following methods is very close. We recommend you choose the method you are the most comfortable with.

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

We will use the keys in the partition features in order to construct a `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_. Let's move to the transformations.

NumPy
-----
The first way is to transform the data into the NumPy arrays. It's an easier option that is commonly used. Feel free to
follow the :doc:`how-to-use-with-numpy` tutorial, especially if you are a beginner.

.. _tensorflow-dataset:

TensorFlow Dataset
------------------
Transform the data to ``TensorFlow Dataset``::

  tf_dataset = partition.to_tf_dataset(columns="img", label_cols="label", batch_size=64,
                                     shuffle=True)
  # Assuming you have defined your model and compiled it
  model.fit(tf_dataset, epochs=20)

TensorFlow Tensors
------------------
Transform the data to the TensorFlow `tf.Tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_ (it's not the TensorFlow dataset)::

  data_tf = partition.with_format("tf")
  # Assuming you have defined your model and compiled it
  model.fit(data_tf["img"], data_tf["label"], epochs=20, batch_size=64)

CNN Keras Model
---------------
Here's a quick example of how you can use that data with a simple CNN model (it assumes you created the TensorFlow
dataset as in the section above, see :ref:`TensorFlow Dataset <tensorflow-dataset>`)::

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
  model.fit(tf_dataset, epochs=20)

