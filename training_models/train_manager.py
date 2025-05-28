import os
import numpy as np
import mlflow.tensorflow
import mlflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from scripts.color_logger import ColorLogger
from scripts.mlflow_manager import MLflowManager, parse_args
from scripts.tensorboard import TensorboardManager
from models_code.metrics.dice_metrics import dice_coefficient, iou_score
from models_code.metrics.classification_metrics import precision, recall, f1_score
from models_code.metrics.custom_losses import FocalLoss, DiceLoss
from config import (
    MODEL_TYPE,
    HEAD_NAMES,
    LOSS_WEIGHTS,
    DATA_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    TRAINING_CONFIG,
    BUILDING_MASK_OPTIONS,
    ROAD_MASK_OPTIONS,
    MASK_PATHS,
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
        self.batch_size = TRAINING_CONFIG["batch_size"]
        self.epochs = TRAINING_CONFIG["epochs"]
        self.learning_rate = TRAINING_CONFIG["learning_rate"]
        
        self.data_loader = DataLoader(
            batch_size=self.batch_size,
            image_size=self.input_shape[:2]
        )
        self.mlflow_manager = None
        self.tensorboard_manager = TensorboardManager(log_dir=self.log_dir)
        self.mask_loader = MaskLoader()

    def get_user_input(self):
        experiment_name = input("Enter experiment name: ")
        run_name = input("Enter model/run name: ")
        output_file = input("Enter output file name (e.g., unet_multi_head.keras): ")
        print("\nAvailable building mask options:")
        for i, option in enumerate(BUILDING_MASK_OPTIONS, 1):
            print(f"{i}. {option}")
        building_mask = BUILDING_MASK_OPTIONS[int(input("Select building mask version (1-2): ")) - 1]
        print("\nAvailable road mask options:")
        for i, option in enumerate(ROAD_MASK_OPTIONS, 1):
            print(f"{i}. {option}")
        road_mask = ROAD_MASK_OPTIONS[int(input("Select road mask version (1-2): ")) - 1]
        MASK_PATHS['buildings'] = building_mask
        MASK_PATHS['roads'] = road_mask
        self.logger.info(f"[TRAIN_MANAGER] MASK_PATHS: {MASK_PATHS}")
        return experiment_name, run_name, output_file, building_mask, road_mask

    def load_data(self):
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")

        train_data = self.data_loader.load(train_dir)
        val_data = self.data_loader.load(val_dir)
        return train_data, val_data

    def train(self):
        args = parse_args()
        experiment_name, run_name, output_file, building_mask, road_mask = self.get_user_input()
        
        self.logger.info("Starting training process...")
        mlflow.tensorflow.autolog(disable=True)
        self.mlflow_manager = MLflowManager(experiment_name, args.description)
        self.mlflow_manager.start_run(run_name=run_name)
        
        train_data, val_data = self.load_data()
        
        self.model = ModelFactory.create_model(MODEL_TYPE)

        building_metrics = [
            'accuracy',
            dice_coefficient,
            iou_score,
            precision,
            recall,
            f1_score
        ]
        
        road_metrics = [
            dice_coefficient,
            iou_score,
            precision,
            recall,
            f1_score
        ]
        
        loss_dict = {
            'head_buildings': DiceLoss(),
            'head_roads': FocalLoss(alpha=0.25, gamma=2.0, from_logits=MASK_PATHS['roads'] == ROAD_MASK_OPTIONS[1])
        }
        
        metrics_dict = {
            'head_buildings': building_metrics,
            'head_roads': road_metrics
        }
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss_dict,
            loss_weights=LOSS_WEIGHTS,
            metrics=metrics_dict
        )
        self.logger.info(f"[TRAIN_MANAGER] Model output names: {self.model.output_names}")
        self.logger.info(f"[TRAIN_MANAGER] Loss dict keys: {list(loss_dict.keys())}")
        self.logger.info(f"[TRAIN_MANAGER] Metrics dict keys: {list(metrics_dict.keys())}")
        
        self.mlflow_manager.log_params({
            "model_type": MODEL_TYPE,
            "input_shape": self.input_shape,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "loss_weights": LOSS_WEIGHTS,
            "building_mask": building_mask,
            "road_mask": road_mask
        })
        
        steps_per_epoch = max(1, train_data.cardinality().numpy() // self.batch_size)
        validation_steps = max(1, val_data.cardinality().numpy() // self.batch_size)
        
        callbacks = [
            self.tensorboard_manager.get_callback(experiment_name=experiment_name),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_dir, f"best_{output_file}"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            try:
                history = self.model.fit(
                    train_data.repeat(),
                    validation_data=val_data,
                    epochs=1,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    callbacks=callbacks
                )
                
                train_metrics = {key: values[-1] for key, values in history.history.items() if values[-1] is not None}

                def safe_fmt(val):
                    return f"{val:.4f}" if isinstance(val, (float, int)) else str(val)

                for head_name in HEAD_NAMES:
                    metrics_str = []
                    for metric in ['loss', 'dice_coefficient', 'iou_score', 'f1_score']:
                        train_val = train_metrics.get(f'head_{head_name}_{metric}', 'N/A')
                        val_val = train_metrics.get(f'val_head_{head_name}_{metric}', 'N/A')
                        metrics_str.append(f"{metric}: {safe_fmt(train_val)}/{safe_fmt(val_val)}")
                    
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.epochs} - {head_name} - " + " | ".join(metrics_str)
                    )
                
                for key, value in train_metrics.items():
                    mlflow.log_metric(key, value, step=epoch)
                    
            except Exception as e:
                self.logger.error(f"Skipping epoch {epoch + 1} due to error: {e}")
                continue
        
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_file)
        self.model.save(output_path)
        
        input_example = np.random.random((1, *self.input_shape)).astype(np.float32)
        self.mlflow_manager.log_model(self.model, model_name=MODEL_TYPE.lower(), input_example=input_example)
        self.mlflow_manager.end_run()
        
        self.logger.info("Training and evaluation process completed.")

if __name__ == "__main__":
    logger = ColorLogger("Main").get_logger()
    try:
        trainer = ModelTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred: {e}")