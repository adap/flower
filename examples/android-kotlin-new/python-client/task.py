import tensorflow as tf
from sklearn.datasets import make_classification

SHAPE = (32, 30)


class LogisticRegressionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(LogisticRegressionModel, self).__init__(**kwargs)

        self.dense_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        outputs = self.dense_layer(inputs)
        return outputs


def get_data():
    X, y = make_classification(
        n_samples=1000,  # 1000 observations
        n_features=30,  # 30 total features
        n_informative=15,  # 15 'useful' features
        n_classes=2,  # binary target/label
        random_state=876,
    )

    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]

    return X_train, y_train, X_test, y_test
