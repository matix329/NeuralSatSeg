from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class UNET:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1, multi_head=False, head_names=None, use_binary_embedding=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.multi_head = multi_head
        self.head_names = head_names
        self.use_binary_embedding = use_binary_embedding
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True
        )

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        if self.use_binary_embedding:
            binary_input = layers.Input(shape=self.input_shape[:2] + (1,))
        x = inputs
        skips = []
        for filters in [64, 128, 256]:
            if self.use_binary_embedding:
                binary_emb = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(binary_input)
                x = layers.Concatenate()([x, binary_emb])
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            skips.append(x)
            x = layers.MaxPooling2D((2, 2))(x)
        b = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        b = layers.BatchNormalization()(b)
        for filters, skip in zip([256, 128, 64], reversed(skips)):
            b = layers.UpSampling2D((2, 2))(b)
            if self.use_binary_embedding:
                binary_emb = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(binary_input)
                b = layers.Concatenate()([b, skip, binary_emb])
            else:
                b = layers.Concatenate()([b, skip])
            b = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(b)
            b = layers.BatchNormalization()(b)
        if self.multi_head and self.head_names:
            outputs = []
            for head_name in self.head_names:
                out = layers.Conv2D(1, (1, 1), activation='sigmoid', name=f'head_{head_name}')(b)
                outputs.append(out)
            if self.use_binary_embedding:
                return models.Model([inputs, binary_input], outputs)
            else:
                return models.Model(inputs, outputs)
        else:
            out = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(b)
            if self.use_binary_embedding:
                return models.Model([inputs, binary_input], out)
            else:
                return models.Model(inputs, out)

    def augment_data(self, train_images, train_masks, batch_size=16):
        image_gen = self.datagen.flow(train_images, batch_size=batch_size, seed=42)
        mask_gen = self.datagen.flow(train_masks, batch_size=batch_size, seed=42)

        return zip(image_gen, mask_gen)