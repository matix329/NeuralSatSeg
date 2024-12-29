from tensorflow.keras.optimizers import Adam
from preprocess import DataLoader
from models_code.unet import UNET
from color_logger import ColorLogger  # Import ColorLogger
import os

class ModelTrainer:
    def __init__(self, train_dir, val_dir, input_shape=(64, 64, 3), num_classes=10, batch_size=32):
        self.logger = ColorLogger("ModelTrainer").get_logger()
        self.train_image_dir = os.path.join(train_dir, "images/train")
        self.train_mask_dir = os.path.join(train_dir, "masks/train")
        self.val_image_dir = os.path.join(val_dir, "images/val")
        self.val_mask_dir = os.path.join(val_dir, "masks/val")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.data_loader = DataLoader(batch_size=batch_size, image_size=input_shape[:2])

    def load_data(self):
        train_data = self.data_loader.load(self.train_image_dir, self.train_mask_dir)
        val_data = self.data_loader.load(self.val_image_dir, self.val_mask_dir)
        return train_data, val_data

    def train(self, output_file="models/model_patched.h5"):
        self.logger.info("Starting training...")
        train_data, val_data = self.load_data()

        model = UNET(input_shape=self.input_shape, num_classes=self.num_classes).build_model()

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            train_data,
            validation_data=val_data,
            epochs=25,
            steps_per_epoch=len(train_data) // self.batch_size,
            validation_steps=len(val_data) // self.batch_size,
        )

        self.logger.info(f"Saving the model to {output_file}...")
        model.save(output_file)
        self.logger.info("Training complete and model saved successfully.")

if __name__ == "__main__":
    output_file = input("Enter the name of the output file (e.g., 'unet_model.keras or unet_1.h5'): ")
    output_path = os.path.join("models", output_file)

    trainer = ModelTrainer(
        train_dir="/Users/matix329/PycharmProjects/NeuralSatSeg/data/processed",
        val_dir="/Users/matix329/PycharmProjects/NeuralSatSeg/data/processed",
        input_shape=(64, 64, 3),
        num_classes=10,
        batch_size=32
    )
    trainer.train(output_file=output_path)