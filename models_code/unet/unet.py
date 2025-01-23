from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models_code.unet.blocks import conv_block, encoder_block, decoder_block

class UNET:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1, activation='leaky_relu', dropout_rate=0.4, output_activation='sigmoid'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        enc_filters = [64, 128, 256]
        encoders = []
        x = inputs
        for filters in enc_filters:
            s, x = encoder_block(x, filters, activation=self.activation, dropout_rate=self.dropout_rate)
            encoders.append(s)

        b = conv_block(x, 512, activation=self.activation, dropout_rate=self.dropout_rate)

        dec_filters = [256, 128, 64]
        for filters, skip in zip(dec_filters, reversed(encoders)):
            b = decoder_block(b, skip, filters, activation=self.activation, dropout_rate=self.dropout_rate)

        outputs = layers.Conv2D(self.num_classes, kernel_size=(1, 1), activation=self.output_activation, padding='same')(b)

        return models.Model(inputs, outputs)

    def augment_data(self, train_images, train_masks, batch_size=16):
        image_gen = self.datagen.flow(train_images, batch_size=batch_size, seed=42)
        mask_gen = self.datagen.flow(train_masks, batch_size=batch_size, seed=42)

        return zip(image_gen, mask_gen)