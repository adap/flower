import tensorflow as tf

SAVED_MODEL_DIR = "saved_model"


class ModelWrapper(tf.Module):
    def __init__(self, model):
        self.model = model
        self.concrete_functions = []

    def train(self, x, y):
        return self.model.train_step((x, y))

    def evaluate(self, x, y):
        return {"accuracy": tf.cast(self.model(x) > 0.5, tf.float32)}

    def layer_count(self):
        return {"count": tf.constant(len(self.model.weights), tf.int32)}

    def parameters(self):
        return {
            f"a{index}": weight.read_value()
            for index, weight in enumerate(self.model.weights)
        }

    def restore(self, **parameters):
        for index, weight in enumerate(self.model.weights):
            parameter = parameters[f"a{index}"]
            weight.assign(parameter)
        assert self.parameters is not None
        return self.parameters()

    def layers(self):
        return {
            f"a{index}": weight.shape for index, weight in enumerate(self.model.weights)
        }

    def add_preprocessing_func(self, func, index):
        tf_func = tf.function(func, input_signature=[tf.TensorSpec([len(index), None])])
        self.concrete_functions.append(tf_func.get_concrete_function())

    def convert_to_tflite(self, tflite_file):
        x_spec = tf.TensorSpec(self.model.input_shape, tf.float32)
        y_spec = tf.TensorSpec(self.model.output_shape, tf.float32)
        train = tf.function(self.train, input_signature=[x_spec, y_spec])
        evaluate = tf.function(self.evaluate, input_signature=[x_spec, y_spec])
        parameters = tf.function(self.parameters, input_signature=[])
        restore = tf.function(self.restore)
        layers = tf.function(self.layers, input_signature=[])
        layer_count = tf.function(self.layer_count, input_signature=[])

        self.concrete_functions.append(train.get_concrete_function())
        self.concrete_functions.append(evaluate.get_concrete_function())
        concrete_parameters = parameters.get_concrete_function()
        self.concrete_functions.append(concrete_parameters)
        init_params = concrete_parameters()
        self.concrete_functions.append(restore.get_concrete_function(**init_params))
        self.concrete_functions.append(layers.get_concrete_function())
        self.concrete_functions.append(layer_count.get_concrete_function())

        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            self.concrete_functions, self.model
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        tflite_model = converter.convert()
        save_tflite_model(tflite_model, tflite_file)
        return tflite_model


def save_tflite_model(tflite_model, tflite_file):
    with open(tflite_file, "wb") as model_file:
        return model_file.write(tflite_model)
