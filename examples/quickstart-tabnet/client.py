import os
import flwr as fl
import tensorflow as tf
import tensorflow_datasets as tfds
import tabnet

train_size = 125
BATCH_SIZE = 50
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def transform(ds):
    features = tf.unstack(ds["features"])
    labels = ds["label"]

    x = dict(zip(col_names, features))
    y = tf.one_hot(labels, 3)
    return x, y


def prepare_iris_dataset():
    ds_full = tfds.load(name="iris", split=tfds.Split.TRAIN)
    ds_full = ds_full.shuffle(150, seed=0)

    ds_train = ds_full.take(train_size)
    ds_train = ds_train.map(transform)
    ds_train = ds_train.batch(BATCH_SIZE)

    ds_test = ds_full.skip(train_size)
    ds_test = ds_test.map(transform)
    ds_test = ds_test.batch(BATCH_SIZE)

    feature_columns = []
    for col_name in col_names:
        feature_columns.append(tf.feature_column.numeric_column(col_name))

    return ds_train, ds_test, feature_columns


ds_train, ds_test, feature_columns = prepare_iris_dataset()
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load TabNet model
model = tabnet.TabNetClassifier(
    feature_columns,
    num_classes=3,
    feature_dim=8,
    output_dim=4,
    num_decision_steps=4,
    relaxation_factor=1.0,
    sparsity_coefficient=1e-5,
    batch_momentum=0.98,
    virtual_batch_size=None,
    norm_type="group",
    num_groups=1,
)
lr = tf.keras.optimizers.schedules.ExponentialDecay(
    0.01, decay_steps=100, decay_rate=0.9, staircase=False
)
optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


# Define Flower client
class TabNetClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(ds_train, epochs=25)
        return model.get_weights(), len(ds_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(ds_test)
        return loss, len(ds_train), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080", client=TabNetClient().to_client()
)
