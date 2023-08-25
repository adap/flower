from tensorflow import keras

def create_model(input_shape, num_classes):
    # CNN Model from (McMahan et. al., 2017) Communication-efficient learning of deep networks from decentralized data
    model = keras.Sequential([
        keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64, (5,5), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
        ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model