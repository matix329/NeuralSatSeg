import os
from tensorflow.keras.optimizers import Adam
from scripts.preprocess import DataLoader
from models_code.unet import UNET
from scripts.color_logger import ColorLogger
from scripts.mlflow_manager import MLflowManager
import numpy as np

class ModelTrainer:
    def __init__(self, train_dir, val_dir, input_shape=(256, 256, 3), num_classes=10, batch_size=32):
        self.logger = ColorLogger("ModelTrainer").get_logger()
        self.train_image_dir = os.path.join(train_dir, "images/train")
        self.train_mask_dir = os.path.join(train_dir, "masks/train")
        self.val_image_dir = os.path.join(val_dir, "images/val")
        self.val_mask_dir = os.path.join(val_dir, "masks/val")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.data_loader = DataLoader(batch_size=batch_size, image_size=input_shape[:2], num_classes=num_classes)
        self.mlflow_manager = None

    def load_data(self):
        train_data = self.data_loader.load(self.train_image_dir, self.train_mask_dir)
        val_data = self.data_loader.load(self.val_image_dir, self.val_mask_dir)
        if train_data.cardinality().numpy() == 0 or val_data.cardinality().numpy() == 0:
            raise ValueError("Training or validation data is empty.")
        return train_data, val_data

    def build_model(self, model_type):
        if model_type.lower() == "unet":
            return UNET(input_shape=self.input_shape, num_classes=self.num_classes).build_model()
        raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, model_type, output_file, experiment_name, run_name, epochs=10):
        self.mlflow_manager = MLflowManager(experiment_name)
        self.mlflow_manager.start_run(run_name=run_name)

        train_data, val_data = self.load_data()
        model = self.build_model(model_type)

        model.compile(
            optimizer=Adam(learning_rate=0.05),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.mlflow_manager.log_params({
            "model_type": model_type,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "epochs": epochs,
            "learning_rate": 0.05
        })

        steps_per_epoch = max(1, train_data.cardinality().numpy() // self.batch_size)
        validation_steps = max(1, val_data.cardinality().numpy() // self.batch_size)

        history = model.fit(
            train_data.repeat(),
            validation_data=val_data.repeat(),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )

        self.mlflow_manager.log_metrics({
            "train_accuracy": max(history.history["accuracy"]),
            "val_accuracy": max(history.history["val_accuracy"]),
            "train_loss": min(history.history["loss"]),
            "val_loss": min(history.history["val_loss"]),
        })

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        model.save(output_file)

        input_example = np.random.random((1, *self.input_shape))
        self.mlflow_manager.log_model(model, model_name=model_type.lower(), input_example=input_example)
        self.mlflow_manager.end_run()

if __name__ == "__main__":
    logger = ColorLogger("Main").get_logger()

    try:
        experiment_name = input("Enter the name of the experiment: ")
        run_name = input("Enter the name of the model/run: ")

        model_type = None
        while model_type not in ["unet", "cnn"]:
            model_type = input("Enter the model type (unet/cnn): ").lower()

        base_output_file = input("Enter the name of the output file (e.g., 'unet_1'): ")
        output_file = f"{base_output_file}.keras"
        output_path = os.path.join("models", output_file)

        trainer = ModelTrainer(
            train_dir="/Users/matix329/PycharmProjects/NeuralSatSeg/data/processed",
            val_dir="/Users/matix329/PycharmProjects/NeuralSatSeg/data/processed"
        )
        trainer.train(model_type=model_type, output_file=output_path, experiment_name=experiment_name, run_name=run_name)
    except Exception as e:
        logger.error(f"An error occurred: {e}")