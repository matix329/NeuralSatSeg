import os
import mlflow.tensorflow
from tensorflow.keras.optimizers import Adam
from scripts.data_loader import DataLoader
from models_code.unet.unet import UNET
from scripts.color_logger import ColorLogger
from scripts.mlflow_manager import MLflowManager
from scripts.tensorboard import TensorboardManager
import numpy as np

class ModelTrainer:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1, batch_size=32):
        self.logger = ColorLogger("ModelTrainer").get_logger()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = os.path.join(project_root, "data/processed/roads")
        self.output_dir = os.path.join(project_root, "output/mlflow_artifacts/models")
        self.train_image_dir = os.path.join(self.data_dir, "processed_images")
        self.train_mask_dir = os.path.join(self.data_dir, "processed_masks")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.data_loader = DataLoader(batch_size=batch_size, image_size=input_shape[:2], num_classes=num_classes)
        self.mlflow_manager = None
        self.tensorboard_manager = TensorboardManager(log_dir=os.path.join(project_root, "data/logs"))

    def validate_directory(self, directory, description):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"{description} directory does not exist: {directory}")
        if not os.listdir(directory):
            raise FileNotFoundError(f"{description} directory is empty: {directory}")

    def load_data(self):
        self.validate_directory(self.train_image_dir, "Training image")
        self.validate_directory(self.train_mask_dir, "Training mask")
        self.logger.info("Loading training data...")
        train_data = self.data_loader.load(self.train_image_dir, self.train_mask_dir)
        return train_data

    def build_model(self, model_type):
        self.logger.info(f"Building model of type: {model_type}")
        supported_models = {
            "unet": lambda: UNET(input_shape=self.input_shape, num_classes=self.num_classes).build_model()
        }
        if model_type.lower() not in supported_models:
            raise ValueError(f"Unsupported model type: {model_type}. Supported models are: {', '.join(supported_models.keys())}")
        model = supported_models[model_type.lower()]()
        #self.logger.debug(f"Model summary:\n{model.summary()}")
        return model

    def train(self, model_type, output_file, experiment_name, run_name, epochs=20):
        self.logger.info("Starting training process...")
        self.logger.info(f"Model type: {model_type}, Output file: {output_file}")
        mlflow.tensorflow.autolog(disable=True)
        self.mlflow_manager = MLflowManager(experiment_name)
        self.mlflow_manager.start_run(run_name=run_name)
        train_data = self.load_data()
        model = self.build_model(model_type)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
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
        self.logger.debug(f"Steps per epoch: {steps_per_epoch}")
        tensorboard_callback = self.tensorboard_manager.get_callback(experiment_name=experiment_name)
        for epoch in range(epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            try:
                history = model.fit(
                    train_data.repeat(),
                    epochs=1,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[tensorboard_callback]
                )
                self.mlflow_manager.log_epoch_metrics(history=history, epoch=epoch)
                self.logger.info(f"Completed epoch {epoch + 1}/{epochs}")
                for key, value in history.history.items():
                    self.logger.debug(f"Epoch {epoch + 1}, Metric {key}: {value[-1]}")
            except Exception as e:
                self.logger.error(f"Skipping epoch {epoch + 1} due to error: {e}")
                continue
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_file)
        self.logger.debug(f"Saving model to: {output_path}")
        model.save(output_path)
        input_example = np.random.random((1, *self.input_shape)).astype(np.float32)
        self.logger.debug(f"Target shape: {train_data.element_spec[1].shape}")
        self.logger.debug(f"Output shape: {model.output_shape}")
        self.mlflow_manager.log_model(model, model_name=model_type.lower(), input_example=input_example)
        self.mlflow_manager.end_run()
        self.logger.info("Training process completed.")

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
        trainer = ModelTrainer(input_shape=(512, 512, 3))
        trainer.train(model_type=model_type, output_file=output_file, experiment_name=experiment_name,
                      run_name=run_name)
    except Exception as e:
        logger.error(f"An error occurred: {e}")