from tensorflow import keras

def prepare_dataset(FEMNIST):
    if FEMNIST == True:
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
  else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

  return x_train, y_train, x_test, y_test, input_shape, num_classes

def partition(x_train, y_train, num_clients, concentration, num_classes):
  dataset = [x_train, y_train]
  partitions, b = create_lda_partitions(dataset, num_partitions=num_clients, concentration=concentration * num_classes, seed=1234)
  return partitions
