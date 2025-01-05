from tensorflow.keras import layers, models
from models_code.unet.blocks import conv_block, encoder_block, decoder_block

class UNET:
    def __init__(self, input_shape=(64, 64, 3), num_classes=10, activation='relu', dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.dropout_rate = dropout_rate

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        s1, p1 = encoder_block(inputs, 32, activation=self.activation, dropout_rate=self.dropout_rate)
        s2, p2 = encoder_block(p1, 64, activation=self.activation, dropout_rate=self.dropout_rate)
        s3, p3 = encoder_block(p2, 128, activation=self.activation, dropout_rate=self.dropout_rate)

        b1 = conv_block(p3, 256, activation=self.activation, dropout_rate=self.dropout_rate)

        d1 = decoder_block(b1, s3, 128, activation=self.activation, dropout_rate=self.dropout_rate)
        d2 = decoder_block(d1, s2, 64, activation=self.activation, dropout_rate=self.dropout_rate)
        d3 = decoder_block(d2, s1, 32, activation=self.activation, dropout_rate=self.dropout_rate)

        outputs = layers.Conv2D(self.num_classes, kernel_size=(1, 1), activation='softmax')(d3)

        return models.Model(inputs, outputs)