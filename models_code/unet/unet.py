from tensorflow.keras import layers, models
from models_code.unet.blocks import conv_block, encoder_block, decoder_block

class UNET:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1, activation='relu', dropout_rate=0.3, output_activation='sigmoid'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        enc_filters = [32, 64, 128]
        encoders = []
        x = inputs
        for filters in enc_filters:
            s, x = encoder_block(x, filters, activation=self.activation, dropout_rate=self.dropout_rate)
            encoders.append(s)

        b = conv_block(x, 256, activation=self.activation, dropout_rate=self.dropout_rate)

        dec_filters = [128, 64, 32]
        for filters, skip in zip(dec_filters, reversed(encoders)):
            b = decoder_block(b, skip, filters, activation=self.activation, dropout_rate=self.dropout_rate)

        outputs = layers.Conv2D(self.num_classes, kernel_size=(1, 1), activation=self.output_activation, padding='same')(b)

        return models.Model(inputs, outputs)