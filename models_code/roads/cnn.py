import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from models_code.roads.metrics import dice_coefficient, iou_score, dice_loss


def get_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1, 0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

def conv_block(x, filters, dropout_rate=0.3):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

def create_road_detection_cnn(input_shape=(640, 640, 3), num_classes=1, compile_model=True, callbacks=False, use_augmentation=True, weighted_bce_weight=0.2, loss_function=None, use_skip_connections=False):
    inputs = layers.Input(input_shape)
    x = inputs
    if use_augmentation:
        x = get_augmentation_layer()(x)
    
    skip_connections = []
    
    x = conv_block(x, 32, 0.3)
    if use_skip_connections:
        skip_connections.append(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 64, 0.3)
    if use_skip_connections:
        skip_connections.append(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 128, 0.3)
    if use_skip_connections:
        skip_connections.append(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 256, 0.4)
    if use_skip_connections:
        skip_connections.append(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 512, 0.4)
    
    if use_skip_connections:
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip_connections.pop()])
        x = conv_block(x, 256, 0.4)
        
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip_connections.pop()])
        x = conv_block(x, 128, 0.3)
        
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip_connections.pop()])
        x = conv_block(x, 64, 0.3)
        
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip_connections.pop()])
        x = conv_block(x, 32, 0.3)
    else:
        x = layers.UpSampling2D(2)(x)
        x = conv_block(x, 256, 0.4)
        x = layers.UpSampling2D(2)(x)
        x = conv_block(x, 128, 0.3)
        x = layers.UpSampling2D(2)(x)
        x = conv_block(x, 64, 0.3)
        x = layers.UpSampling2D(2)(x)
        x = conv_block(x, 32, 0.3)
    
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    
    if compile_model:
        if loss_function is None:
            loss_function = combined_loss
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=loss_function,
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