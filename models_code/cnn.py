from tensorflow.keras import layers, Model

class CNN:
    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = layers.Input(self.input_shape)

        x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        return Model(inputs, outputs)