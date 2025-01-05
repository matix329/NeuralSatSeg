from tensorflow.keras import layers

def conv_block(inputs, filters, activation='relu', dropout_rate=0.3):
    x = layers.Conv2D(filters, kernel_size=(3, 3), activation=activation, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, kernel_size=(3, 3), activation=activation, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def encoder_block(inputs, filters, activation='relu', dropout_rate=0.3):
    x = conv_block(inputs, filters, activation=activation, dropout_rate=dropout_rate)
    p = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, filters, activation='relu', dropout_rate=0.3):
    x = layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(inputs)
    x = layers.concatenate([x, skip_features])
    x = conv_block(x, filters, activation=activation, dropout_rate=dropout_rate)
    return x