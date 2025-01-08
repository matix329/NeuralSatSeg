import os
import mlflow.tensorflow
from tensorflow.keras.optimizers import Adam
from scripts.preprocess import DataLoader
from models_code.unet.unet import UNET
from scripts.color_logger import ColorLogger
from scripts.mlflow_manager import MLflowManager
from scripts.tensorboard import TensorboardManager
import numpy as np

class ModelTrainer:
    def __init__(self, input_shape=(128, 128, 3), num_classes=10, batch_size=32):
        self.logger = ColorLogger("ModelTrainer").get_logger()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(base_dir, "../data/processed")
        self.output_dir = os.path.join(base_dir, "../mlflow_artifacts/models")

        self.train_image_dir = os.path.join(self.data_dir, "images/train")
        self.train_mask_dir = os.path.join(self.data_dir, "masks/train")
        self.val_image_dir = os.path.join(self.data_dir, "images/val")
        self.val_mask_dir = os.path.join(self.data_dir, "masks/val")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.data_loader = DataLoader(batch_size=batch_size, image_size=input_shape[:2], num_classes=num_classes)
        self.mlflow_manager = None
        self.tensorboard_manager = TensorboardManager(log_dir=os.path.join(self.output_dir, "logs"))

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

    def train(self, model_type, output_file, experiment_name, run_name, epochs=5):
        mlflow.tensorflow.autolog(disable=True)

        self.mlflow_manager = MLflowManager(experiment_name)
        self.mlflow_manager.start_run(run_name=run_name)

        train_data, val_data = self.load_data()
        model = self.build_model(model_type)

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.mlflow_manager.log_params({
            "model_type": model_type,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "epochs": epochs,
            "learning_rate": 0.001
        })

        steps_per_epoch = max(1, train_data.cardinality().numpy() // self.batch_size)
        validation_steps = max(1, val_data.cardinality().numpy() // self.batch_size)

        tensorboard_callback = self.tensorboard_manager.get_callback(experiment_name=experiment_name)

        for epoch in range(epochs):
            history = model.fit(
                train_data.repeat(),
                validation_data=val_data.repeat(),
                epochs=1,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=[tensorboard_callback]
            )
            self.mlflow_manager.log_epoch_metrics(
                history=history,
                epoch=epoch
            )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_file)
        model.save(output_path)

        input_example = np.random.random((1, *self.input_shape)).astype(np.float32)
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

        trainer = ModelTrainer()
        trainer.train(model_type=model_type, output_file=output_file, experiment_name=experiment_name, run_name=run_name)
    except Exception as e:
        logger.error(f"An error occurred: {e}")