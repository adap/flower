Use with TensorFlow
===================

Let's integrate ``flwr-datasets`` with TensorFlow. We show you three ways how to transform the data into the formats
that ``TensorFlow``'s models expect. Please note that especially for the smaller datasets the performance of all of the
following methods is very close therefore we recommend for you to choose the method you are the most comfortable using.

Numpy
-----
The first way is to transform the data into the numpy arrays. It's an easier option that is commonly used. Feel free to
follow the :doc:`how-to-use-with-numpy` tutorial especially if you are a beginner.

TensorFlow Tensors
------------------
Change the data type to TensorFlow Tensors (it's not the TensorFlow dataset).

Standard setup::

  from flwr_datasets import FederatedDataset
  fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10}) # Divide "train" into 10 partitions
  partition = fds.load_partition(0, "train") # Load 0-th index partition
  centralized_dataset = fds.load_full("test") # Load dataset for centralized dataset

Transformation to the TensorFlow Tensors ::

  data_tf = partition.with_format("tf")
  # Assuming you have defined your model and compiled it
  model.fit(data_tf["img"], data_tf["label"], epochs=20, batch_size=64)

TensorFlow Dataset
------------------
Work with ``TensorFlow Dataset`` abstraction.

Standard setup::

  from flwr_datasets import FederatedDataset
  fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10}) # Divide "train" into 10 partitions
  partition = fds.load_partition(0, "train") # Load 0-th index partition
  centralized_dataset = fds.load_full("test") # Load dataset for centralized dataset

Transformation to the TensorFlow Dataset::

  tf_dataset = partition.to_tf_dataset(columns="img", label_cols="label", batch_size=64, shuffle=True)
  # Assuming you have defined your model and compiled it
  model.fit(tf_dataset, epochs=20)



CNN Keras Model
---------------
Here's a quick example how you can use that data with a simple CNN model::

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

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=20, batch_size=64)


