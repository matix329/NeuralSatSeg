from tensorflow.keras import layers, Model

class UNET:
    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = layers.Input(self.input_shape)

        # Enkoder
        c1, p1 = self.conv_block(inputs, 64)
        c2, p2 = self.conv_block(p1, 128)
        c3, p3 = self.conv_block(p2, 256)
        c4, p4 = self.conv_block(p3, 512)

        # Bottleneck
        bottleneck = self.conv_block(p4, 1024, pool=False)

        # Global pooling i klasyfikacja
        x = layers.GlobalAveragePooling2D()(bottleneck)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        return Model(inputs, outputs)

    def conv_block(self, inputs, filters, pool=True):
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        if pool:
            return x, layers.MaxPooling2D((2, 2))(x)
        return x