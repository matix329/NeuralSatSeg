import tensorflow as tf
from tensorflow.keras import layers, Model

def create_unet(input_shape=(650, 650, 3), num_classes=1):
    inputs = layers.Input(input_shape)

    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = layers.Conv2D(512, 3, activation="relu", padding="same")(conv4)

    up3 = layers.UpSampling2D(size=(2, 2))(conv4)
    up3 = layers.Conv2D(256, 2, padding="same")(up3)
    up3 = layers.Resizing(162, 162)(up3)
    up3 = layers.concatenate([up3, conv3])
    conv5 = layers.Conv2D(256, 3, activation="relu", padding="same")(up3)
    conv5 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv5)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv5)
    up2 = layers.Conv2D(128, 2, padding="same")(up2)
    up2 = layers.Resizing(325, 325)(up2)
    up2 = layers.concatenate([up2, conv2])
    conv6 = layers.Conv2D(128, 3, activation="relu", padding="same")(up2)
    conv6 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv6)
    
    up1 = layers.UpSampling2D(size=(2, 2))(conv6)
    up1 = layers.Conv2D(64, 2, padding="same")(up1)
    up1 = layers.Resizing(650, 650)(up1)
    up1 = layers.concatenate([up1, conv1])
    conv7 = layers.Conv2D(64, 3, activation="relu", padding="same")(up1)
    conv7 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv7)
    
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model 