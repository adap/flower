from tensorflow import keras
import keras_cv
from keras_cv.models import ResNet18Backbone, ImageClassifier


class ResNetModelBuilder:
    def __init__(
        self,
        lr=0.001,
        include_rescaling=False,
        num_classes=10,
        weights=None,
        net="ResNet50",
        input_shape=(None, None, 3),
    ):
        self.lr = lr
        self.num_classes = num_classes
        self.weights = weights
        self.net = net
        self.input_shape = input_shape
        self.include_rescaling = include_rescaling

    def run(self):
        if self.net == "ResNet18":
            backbone = ResNet18Backbone()
            backbone.layers[2].strides = (1, 1)
            # print(backbone.layers[2].get_config())
            model = ImageClassifier(backbone=backbone, num_classes=self.num_classes)
        elif self.net == "ResNet50":
            backbone = keras_cv.models.ResNet50V2Backbone()
            model = ImageClassifier(
                backbone=backbone,
                num_classes=self.num_classes,
            )
        else:
            fn = getattr(keras_cv.models, self.net)
            model = fn(
                include_rescaling=self.include_rescaling,
                include_top=True,
                weights=self.weights,
                classes=self.num_classes,
                input_shape=self.input_shape,
            )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(self.lr),
            metrics=["accuracy"],
        )
        return model


if __name__ == "__main__":
    import numpy as np

    model = ResNetModelBuilder(net="ResNet50").run()
    example_input = np.random.rand(2, 32, 32, 3)  # 2 images, 32x32 pixels, 3 channels
    out = model(example_input)
    print("output tensor shape:", out.shape)
