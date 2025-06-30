import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def tversky_loss(alpha=0.3, beta=0.7, smooth=1.0):
    def loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        true_pos = K.sum(y_true_f * y_pred_f)
        false_neg = K.sum(y_true_f * (1 - y_pred_f))
        false_pos = K.sum((1 - y_true_f) * y_pred_f)
        return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return loss

def create_unet(input_shape=(650, 650, 3), num_classes=1):
    inputs = layers.Input(input_shape)

    conv1 = layers.Conv2D(64, 3, padding="same")(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)
    conv1 = layers.Conv2D(64, 3, padding="same")(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, padding="same")(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)
    conv2 = layers.Conv2D(128, 3, padding="same")(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, padding="same")(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)
    conv3 = layers.Conv2D(256, 3, padding="same")(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, padding="same")(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation("relu")(conv4)
    conv4 = layers.Conv2D(512, 3, padding="same")(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation("relu")(conv4)

    up3 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(conv4)
    up3 = layers.BatchNormalization()(up3)
    up3 = layers.Activation("relu")(up3)
    up3 = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([up3, conv3])
    up3 = layers.concatenate([up3, conv3])
    conv5 = layers.Conv2D(256, 3, padding="same")(up3)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    conv5 = layers.Conv2D(256, 3, padding="same")(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    
    up2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(conv5)
    up2 = layers.BatchNormalization()(up2)
    up2 = layers.Activation("relu")(up2)
    up2 = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([up2, conv2])
    up2 = layers.concatenate([up2, conv2])
    conv6 = layers.Conv2D(128, 3, padding="same")(up2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation("relu")(conv6)
    conv6 = layers.Conv2D(128, 3, padding="same")(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation("relu")(conv6)
    
    up1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(conv6)
    up1 = layers.BatchNormalization()(up1)
    up1 = layers.Activation("relu")(up1)
    up1 = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([up1, conv1])
    up1 = layers.concatenate([up1, conv1])
    conv7 = layers.Conv2D(64, 3, padding="same")(up1)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation("relu")(conv7)
    conv7 = layers.Conv2D(64, 3, padding="same")(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation("relu")(conv7)
    
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tversky_loss(alpha=0.3, beta=0.7),
        metrics=[
            dice_coefficient,
            iou_score,
            tf.keras.metrics.BinaryAccuracy(threshold=0.3)
        ]
    )
    
    return model

def create_unet_graph(input_shape=(650, 650, 3)):
    inputs = layers.Input(input_shape)

    conv1 = layers.Conv2D(64, 3, padding="same")(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)
    conv1 = layers.Conv2D(64, 3, padding="same")(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, padding="same")(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)
    conv2 = layers.Conv2D(128, 3, padding="same")(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, padding="same")(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)
    conv3 = layers.Conv2D(256, 3, padding="same")(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, padding="same")(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation("relu")(conv4)
    conv4 = layers.Conv2D(512, 3, padding="same")(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation("relu")(conv4)

    up3 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(conv4)
    up3 = layers.BatchNormalization()(up3)
    up3 = layers.Activation("relu")(up3)
    up3 = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([up3, conv3])
    up3 = layers.concatenate([up3, conv3])
    conv5 = layers.Conv2D(256, 3, padding="same")(up3)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    conv5 = layers.Conv2D(256, 3, padding="same")(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    
    up2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(conv5)
    up2 = layers.BatchNormalization()(up2)
    up2 = layers.Activation("relu")(up2)
    up2 = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([up2, conv2])
    up2 = layers.concatenate([up2, conv2])
    conv6 = layers.Conv2D(128, 3, padding="same")(up2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation("relu")(conv6)
    conv6 = layers.Conv2D(128, 3, padding="same")(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation("relu")(conv6)
    
    up1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(conv6)
    up1 = layers.BatchNormalization()(up1)
    up1 = layers.Activation("relu")(up1)
    up1 = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([up1, conv1])
    up1 = layers.concatenate([up1, conv1])
    conv7 = layers.Conv2D(64, 3, padding="same")(up1)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation("relu")(conv7)
    conv7 = layers.Conv2D(64, 3, padding="same")(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation("relu")(conv7)
    
    outputs = layers.Conv2D(3, 1, activation="sigmoid")(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["accuracy"]
    )
    
    return model 