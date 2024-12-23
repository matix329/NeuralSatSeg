from tensorflow.keras import layers, models

def conv_block(inputs, filters):
    x = layers.Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def encoder_block(inputs, filters):
    x = conv_block(inputs, filters)
    p = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, filters):
    x = layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(inputs)
    x = layers.concatenate([x, skip_features])
    x = conv_block(x, filters)
    return x

class UNET:
    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        s1, p1 = encoder_block(inputs, 32)
        s2, p2 = encoder_block(p1, 64)
        s3, p3 = encoder_block(p2, 128)

        b1 = conv_block(p3, 256)

        d1 = decoder_block(b1, s3, 128)
        d2 = decoder_block(d1, s2, 64)
        d3 = decoder_block(d2, s1, 32)

        outputs = layers.Conv2D(self.num_classes, kernel_size=(1, 1), activation='softmax')(d3)

        return models.Model(inputs, outputs)