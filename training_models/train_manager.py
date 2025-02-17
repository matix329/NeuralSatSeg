import os
import numpy as np
import mlflow.tensorflow
import mlflow
from tensorflow.keras.optimizers import Adam
from scripts.data_loader import DataLoader
from models_code.unet.unet import UNET
from scripts.color_logger import ColorLogger
from scripts.mlflow_manager import MLflowManager
from scripts.tensorboard import TensorboardManager
from config import CONFIG

class ModelTrainer:
    def __init__(self):
        self.logger = ColorLogger("ModelTrainer").get_logger()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.train_data_dir = os.path.join(project_root, "data/processed/train/roads")
        self.test_data_dir = os.path.join(project_root, "data/processed/test/roads")
        self.output_dir = os.path.join(project_root, "output/mlflow_artifacts/models")
        self.train_image_dir = os.path.join(self.train_data_dir, "processed_images")
        self.train_mask_dir = os.path.join(self.train_data_dir, "processed_masks")
        self.test_image_dir = os.path.join(self.test_data_dir, "processed_images")
        self.input_shape = CONFIG["input_shape"]
        self.num_classes = CONFIG["num_classes"]
        self.batch_size = CONFIG["batch_size"]
        self.epochs = CONFIG["epochs"]
        self.learning_rate = CONFIG["learning_rate"]
        self.data_loader = DataLoader(batch_size=self.batch_size, image_size=self.input_shape[:2], num_classes=self.num_classes)
        self.mlflow_manager = None
        self.tensorboard_manager = TensorboardManager(log_dir=os.path.join(project_root, "data/logs"))

    def display_training_params(self):
        print("\n=== Training Parameters ===")
        for key, value in CONFIG.items():
            print(f"{key}: {value}")
        print("==========================")

    def confirm_training(self):
        self.display_training_params()
        confirm = input("Do you want to proceed with these settings? (y/n): ").strip().lower()
        return confirm == "y"

    def validate_directory(self, directory, description):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"{description} directory does not exist: {directory}")
        if not os.listdir(directory):
            raise FileNotFoundError(f"{description} directory is empty: {directory}")

    def load_train_data(self):
        self.validate_directory(self.train_image_dir, "Training image")
        self.validate_directory(self.train_mask_dir, "Training mask")
        self.logger.info("Loading training data...")
        return self.data_loader.load(self.train_image_dir, self.train_mask_dir)

    def load_test_data(self):
        self.validate_directory(self.test_image_dir, "Test image")
        self.logger.info("Loading test data...")
        return self.data_loader.load(self.test_image_dir, None)

    def build_model(self, model_type):
        self.logger.info(f"Building model of type: {model_type}")
        supported_models = {
            "unet": lambda: UNET(input_shape=self.input_shape, num_classes=self.num_classes).build_model()
        }
        if model_type.lower() not in supported_models:
            raise ValueError(f"Unsupported model type: {model_type}. Supported models are: {', '.join(supported_models.keys())}")
        return supported_models[model_type.lower()]()

    def train(self, model_type, output_file, experiment_name, run_name):
        if not self.confirm_training():
            self.logger.info("Training cancelled.")
            return

        self.logger.info("Starting training process...")
        mlflow.tensorflow.autolog(disable=True)
        self.mlflow_manager = MLflowManager(experiment_name)
        self.mlflow_manager.start_run(run_name=run_name)

        train_data = self.load_train_data()
        test_images = self.load_test_data()

        model = self.build_model(model_type)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        actual_learning_rate = model.optimizer.learning_rate.numpy()

        self.mlflow_manager.log_params({
            "model_type": model_type,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": actual_learning_rate
        })

        steps_per_epoch = max(1, train_data.cardinality().numpy() // self.batch_size)
        tensorboard_callback = self.tensorboard_manager.get_callback(experiment_name=experiment_name)

        for epoch in range(self.epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.epochs}")
            try:
                history = model.fit(
                    train_data.repeat(),
                    epochs=1,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[tensorboard_callback]
                )
                train_metrics = {key: values[-1] for key, values in history.history.items() if values[-1] is not None}

                predictions = model.predict(test_images, verbose=0)
                test_accuracy = np.mean(np.round(predictions).flatten()) if predictions is not None else "N/A"

                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"train_loss: {train_metrics.get('loss', 'N/A'):.4f}, train_accuracy: {train_metrics.get('accuracy', 'N/A'):.4f}, "
                    f"test_accuracy: {test_accuracy}"
                )

                for key, value in train_metrics.items():
                    mlflow.log_metric(f"train_{key}", value, step=epoch)
                mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

            except Exception as e:
                self.logger.error(f"Skipping epoch {epoch + 1} due to error: {e}")
                continue

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_file)
        model.save(output_path)

        input_example = np.random.random((1, *self.input_shape)).astype(np.float32)
        self.mlflow_manager.log_model(model, model_name=model_type.lower(), input_example=input_example)
        self.mlflow_manager.end_run()
        self.logger.info("Training and evaluation process completed.")

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
        trainer.train(model_type=model_type, output_file=output_file, experiment_name=experiment_name,
                      run_name=run_name)
    except Exception as e:
        logger.error(f"An error occurred: {e}")