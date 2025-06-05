import tensorflow as tf
from tensorflow.keras import layers, Model

def create_cnn(input_shape=(650, 650, 3), num_classes=1):
    inputs = layers.Input(input_shape)
    
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(512, 3, activation="relu", padding="same")(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model 