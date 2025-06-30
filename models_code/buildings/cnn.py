import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters, kernel_size=3, dilation=1, dropout=0.2):
    x = layers.Conv2D(filters, kernel_size, padding="same", dilation_rate=dilation)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", dilation_rate=dilation)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout)(x)
    return x

def create_cnn(input_shape=(650, 650, 3), num_classes=1):
    inputs = layers.Input(input_shape)

    x = conv_block(inputs, 64)
    x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, 128)
    x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, 256)
    x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, 512)
    x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, 1024, dilation=2)
    x = layers.MaxPooling2D(2)(x)

    b1 = conv_block(x, 1024, dilation=2, dropout=0.3)
    b2 = conv_block(x, 1024, dilation=4, dropout=0.3)
    b3 = conv_block(x, 1024, dilation=8, dropout=0.3)
    x = layers.Add()([b1, b2, b3])

    x = layers.Conv2DTranspose(512, 3, strides=2, padding="same")(x)
    x = conv_block(x, 512)
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = conv_block(x, 256)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = conv_block(x, 128)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = conv_block(x, 64)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = conv_block(x, 32)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(num_classes, 1, activation=activation, padding="same")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def deep_cnn_body(*args, **kwargs):
    return create_cnn(*args, **kwargs) 