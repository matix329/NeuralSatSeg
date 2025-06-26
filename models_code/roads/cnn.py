import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from models_code.roads.metrics import dice_coefficient, iou_score, dice_loss

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    return x


def crop_to_match(skip, up):
    def crop(inputs):
        skip, up = inputs
        sh = tf.shape(skip)[1]
        sw = tf.shape(skip)[2]
        uh = tf.shape(up)[1]
        uw = tf.shape(up)[2]
        crop_h = sh - uh
        crop_w = sw - uw
        crop_h = tf.maximum(crop_h, 0)
        crop_w = tf.maximum(crop_w, 0)
        cropped = skip[:, :sh-crop_h, :sw-crop_w, :]
        return cropped[:, :uh, :uw, :]
    return layers.Lambda(crop, output_shape=lambda s: s[1])([skip, up])


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    return bce + dsc

def create_road_detection_cnn(input_shape=(640, 640, 3), num_classes=1, compile_model=True, callbacks=False):
    inputs = layers.Input(input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D(2)(c4)

    bn = conv_block(p4, 512)

    u1 = layers.UpSampling2D(2)(bn)
    u1 = layers.Conv2D(256, 3, padding='same')(u1)
    c4_cropped = crop_to_match(c4, u1)
    u1 = layers.Concatenate()([u1, c4_cropped])
    c5 = conv_block(u1, 256)

    u2 = layers.UpSampling2D(2)(c5)
    u2 = layers.Conv2D(128, 3, padding='same')(u2)
    c3_cropped = crop_to_match(c3, u2)
    u2 = layers.Concatenate()([u2, c3_cropped])
    c6 = conv_block(u2, 128)

    u3 = layers.UpSampling2D(2)(c6)
    u3 = layers.Conv2D(64, 3, padding='same')(u3)
    c2_cropped = crop_to_match(c2, u3)
    u3 = layers.Concatenate()([u3, c2_cropped])
    c7 = conv_block(u3, 64)

    u4 = layers.UpSampling2D(2)(c7)
    u4 = layers.Conv2D(32, 3, padding='same')(u4)
    c1_cropped = crop_to_match(c1, u4)
    u4 = layers.Concatenate()([u4, c1_cropped])
    c8 = conv_block(u4, 32)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(c8)

    model = Model(inputs, outputs)

    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=combined_loss,
            metrics=[dice_coefficient, iou_score, tf.keras.metrics.BinaryAccuracy()]
        )
    if callbacks:
        cb = [
            ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.5, patience=4, min_lr=1e-6, mode='max', verbose=1),
            EarlyStopping(monitor='val_dice_coefficient', patience=10, mode='max', restore_best_weights=True, verbose=1)
        ]
        return model, cb
    return model

create_cnn = create_road_detection_cnn