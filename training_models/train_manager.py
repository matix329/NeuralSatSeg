import os
import numpy as np
import mlflow.tensorflow
import mlflow
from tensorflow.keras.optimizers import Adam
from scripts.color_logger import ColorLogger
from scripts.mlflow_manager import MLflowManager
from scripts.tensorboard import TensorboardManager
from models_code.metrics.dice_metrics import dice_coefficient, iou_score
from models_code.metrics.classification_metrics import precision, recall, f1_score
from config import (
    BUILDING_MASK_OPTIONS,
    ROAD_MASK_OPTIONS,
    SUPPORTED_MODELS,
    DATA_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    TRAINING_CONFIG
)
from data_loader import DataLoader
from model_factory import ModelFactory
from mask_loader import MaskLoader

class ModelTrainer:
    def __init__(self):
        self.logger = ColorLogger("ModelTrainer").get_logger()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = os.path.join(self.project_root, DATA_DIR)
        self.output_dir = os.path.join(self.project_root, OUTPUT_DIR)
        self.log_dir = os.path.join(self.project_root, LOG_DIR)
        
        self.input_shape = TRAINING_CONFIG["input_shape"]
        self.num_classes = TRAINING_CONFIG["num_classes"]
        self.batch_size = TRAINING_CONFIG["batch_size"]
        self.epochs = TRAINING_CONFIG["epochs"]
        self.learning_rate = TRAINING_CONFIG["learning_rate"]
        
        self.data_loader = DataLoader(
            batch_size=self.batch_size,
            image_size=self.input_shape[:2],
            num_classes=self.num_classes
        )
        self.mlflow_manager = None
        self.tensorboard_manager = TensorboardManager(log_dir=self.log_dir)
        self.mask_loader = MaskLoader()

    def get_user_input(self):
        experiment_name = input("Enter experiment name: ")
        
        print("\nAvailable building mask options:")
        for i, option in enumerate(BUILDING_MASK_OPTIONS, 1):
            print(f"{i}. {option}")
        building_mask = BUILDING_MASK_OPTIONS[int(input("Select building mask version (1-2): ")) - 1]
        
        print("\nAvailable road mask options:")
        for i, option in enumerate(ROAD_MASK_OPTIONS, 1):
            print(f"{i}. {option}")
        road_mask = ROAD_MASK_OPTIONS[int(input("Select road mask version (1-2): ")) - 1]
        
        run_name = input("Enter model/run name: ")
        
        print("\nAvailable model types:")
        for i, model in enumerate(SUPPORTED_MODELS, 1):
            print(f"{i}. {model.upper()}")
        model_type = SUPPORTED_MODELS[int(input("Select model type (1-2): ")) - 1]
        
        output_file = input("Enter output file name (e.g., cnn_buildings_graph.keras): ")
        
        description = f"Experiment: {experiment_name}\n" \
                     f"Model: {model_type}\n" \
                     f"Building mask: {building_mask}\n" \
                     f"Road mask: {road_mask}"
        
        return experiment_name, building_mask, road_mask, run_name, model_type, output_file, description

    def load_data(self, mask_type):
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        data_type = "buildings" if "buildings" in mask_type else "roads"
        
        train_images = os.path.join(train_dir, data_type, "images")
        train_masks = os.path.join(train_dir, data_type, mask_type)
        val_images = os.path.join(val_dir, data_type, "images")
        val_masks = os.path.join(val_dir, data_type, mask_type)
        
        if not os.path.exists(train_images):
            raise FileNotFoundError(f"Training images directory not found: {train_images}")
        if not os.path.exists(train_masks):
            raise FileNotFoundError(f"Training masks directory not found: {train_masks}")
        if not os.path.exists(val_images):
            raise FileNotFoundError(f"Validation images directory not found: {val_images}")
        if not os.path.exists(val_masks):
            raise FileNotFoundError(f"Validation masks directory not found: {val_masks}")
        
        train_data = self.data_loader.load(train_images, train_masks)
        val_data = self.data_loader.load(val_images, val_masks)
        
        return train_data, val_data

    def train(self):
        experiment_name, building_mask, road_mask, run_name, model_type, output_file, description = self.get_user_input()
        
        self.logger.info("Starting training process...")
        mlflow.tensorflow.autolog(disable=True)
        self.mlflow_manager = MLflowManager(experiment_name)
        self.mlflow_manager.start_run(run_name=run_name)
        
        mlflow.set_tag("description", description)
        
        train_data, val_data = self.load_data(building_mask if "buildings" in building_mask else road_mask)
        
        model = ModelFactory.create_model(model_type)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", dice_coefficient, iou_score, precision, recall, f1_score]
        )
        
        self.mlflow_manager.log_params({
            "model_type": model_type,
            "building_mask": building_mask,
            "road_mask": road_mask,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate
        })
        
        steps_per_epoch = max(1, train_data.cardinality().numpy() // self.batch_size)
        tensorboard_callback = self.tensorboard_manager.get_callback(experiment_name=experiment_name)
        
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            try:
                history = model.fit(
                    train_data.repeat(),
                    validation_data=val_data,
                    epochs=1,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[tensorboard_callback]
                )
                
                train_metrics = {key: values[-1] for key, values in history.history.items() if values[-1] is not None}
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"train_loss: {train_metrics.get('loss', 'N/A'):.4f}, "
                    f"train_accuracy: {train_metrics.get('accuracy', 'N/A'):.4f}, "
                    f"val_loss: {train_metrics.get('val_loss', 'N/A'):.4f}, "
                    f"val_accuracy: {train_metrics.get('val_accuracy', 'N/A'):.4f}"
                )
                
                for key, value in train_metrics.items():
                    mlflow.log_metric(key, value, step=epoch)
                    
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
        trainer = ModelTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred: {e}")