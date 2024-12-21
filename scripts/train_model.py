from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from models_code import UNET, CNN

class ModelTrainer:
    def __init__(self, model_name, train_data_path, val_data_path, input_shape=(64, 64, 3), num_classes=10):
        self.model_name = model_name
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.input_shape = input_shape
        self.num_classes = num_classes

    def load_data(self):
        train_data = image_dataset_from_directory(
            self.train_data_path,
            shuffle=True,
            batch_size=32,
            image_size=self.input_shape[:2],
            label_mode="categorical"
        )
        val_data = image_dataset_from_directory(
            self.val_data_path,
            shuffle=False,
            batch_size=32,
            image_size=self.input_shape[:2],
            label_mode="categorical"
        )
        return train_data, val_data

    def get_model(self):
        if self.model_name == "unet":
            return UNET(input_shape=self.input_shape, num_classes=self.num_classes).build_model()
        elif self.model_name == "cnn":
            return CNN(input_shape=self.input_shape, num_classes=self.num_classes).build_model()
        else:
            raise ValueError(f"Model {self.model_name} not supported!")

    def train(self):
        train_data, val_data = self.load_data()
        model = self.get_model()

        model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_data, validation_data=val_data, epochs=25)
        model.save(f"models/{self.model_name}_model.h5")


if __name__ == "__main__":
    trainer = ModelTrainer(
        model_name="unet",  # Zmień na "cnn" dla prostego CNN
        train_data_path="/Users/matix329/PycharmProjects/NeuralSatSeg/data/processed/train",
        val_data_path="/Users/matix329/PycharmProjects/NeuralSatSeg/data/processed/val",
        input_shape=(64, 64, 3),
        num_classes=10
    )
    trainer.train()