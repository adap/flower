import tensorflow as tf


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        # Define layers in the constructor
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        # Define the forward pass
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)


def load_data(partition_id: int, num_partitions: int):

    return trainloader, testloader


def train(net, trainloader, valloader, epochs, device):


def test(net, testloader, device):

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)