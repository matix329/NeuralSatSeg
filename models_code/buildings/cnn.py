import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def create_cnn(input_shape=(650, 650, 3), num_classes=1):
    inputs = layers.Input(input_shape)
    
    x = conv_block(inputs, 32)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 64)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 128)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 256)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 512)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 1024)
    
    x = layers.UpSampling2D(2)(x)
    x = conv_block(x, 512)
    
    x = layers.UpSampling2D(2)(x)
    x = conv_block(x, 256)
    
    x = layers.UpSampling2D(2)(x)
    x = conv_block(x, 128)
    
    x = layers.UpSampling2D(2)(x)
    x = conv_block(x, 64)
    
    x = layers.UpSampling2D(2)(x)
    x = conv_block(x, 32)
    
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(num_classes, 1, activation=activation, padding="same")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model 