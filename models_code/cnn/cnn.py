from tensorflow.keras import layers, models

class CNN:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1, multi_head=False, head_names=None, use_binary_embedding=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.multi_head = multi_head
        self.head_names = head_names
        self.use_binary_embedding = use_binary_embedding

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        if self.use_binary_embedding:
            binary_input = layers.Input(shape=self.input_shape[:2] + (1,))
            binary_emb = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(binary_input)
            x = layers.Concatenate()([inputs, binary_emb])
        else:
            x = inputs
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        if self.multi_head and self.head_names:
            outputs = []
            for head_name in self.head_names:
                out = layers.Conv2D(1, (1, 1), activation='sigmoid', name=f'head_{head_name}')(x)
                outputs.append(out)
            if self.use_binary_embedding:
                return models.Model([inputs, binary_input], outputs)
            else:
                return models.Model(inputs, outputs)
        else:
            x = layers.Flatten()(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            if self.use_binary_embedding:
                return models.Model([inputs, binary_input], outputs)
            else:
                return models.Model(inputs, outputs)