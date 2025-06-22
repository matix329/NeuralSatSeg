import os
import numpy as np
import mlflow.tensorflow
import mlflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from scripts.color_logger import ColorLogger
from scripts.mlflow_manager import MLflowManager
from scripts.tensorboard import TensorboardManager
from models_code.roads.metrics import dice_coefficient as roads_dice_coefficient
from models_code.roads.metrics import iou_score as roads_iou_score
from models_code.buildings.metrics import dice_coefficient as buildings_dice_coefficient
from models_code.buildings.metrics import iou_score as buildings_iou_score
from training_models.buildings.config import REDUCED_DATASET_SIZE
from scripts.epoch_logger_callback import EpochLogger

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

class ModelTrainer:
    def __init__(self):
        self.logger = ColorLogger("ModelTrainer").get_logger()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.output_dir = os.path.join(self.project_root, "output/mlflow_artifacts/models")
        self.log_dir = os.path.join(self.project_root, "logs")
        self.mlflow_manager = None
        self.tensorboard_manager = TensorboardManager(log_dir=self.log_dir)
        self.input_shape = None
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None

    def get_user_input(self):
        print("\nSelect model type to train:")
        print("1. Roads")
        print("2. Buildings")
        model_type = int(input("Choose option (1-2): "))
        if model_type == 1:
            from training_models.roads.config import TRAINING_CONFIG as SELECTED_CONFIG
        else:
            from training_models.buildings.config import TRAINING_CONFIG as SELECTED_CONFIG
        self.input_shape = SELECTED_CONFIG["input_shape"]
        self.batch_size = SELECTED_CONFIG["batch_size"]
        self.epochs = SELECTED_CONFIG["epochs"]
        self.learning_rate = SELECTED_CONFIG["learning_rate"]
        
        print("\nSelect model architecture:")
        print("1. UNet (default)")
        print("2. CNN")
        architecture = int(input("Choose option (1-2): "))
        
        use_reduced_dataset = False
        if model_type == 2:
            print("\nUse reduced dataset to match roads size? (y/n):")
            use_reduced_dataset = input().lower() == 'y'
        
        if model_type == 1:
            print("\nSelect road mask type:")
            print("1. Binary")
            print("2. Graph")
            mask_type = int(input("Choose option (1-2): "))
            mask_type = "binary" if mask_type == 1 else "graph"
        else:
            print("\nSelect building mask type:")
            print("1. Original")
            print("2. Eroded")
            mask_type = int(input("Choose option (1-2): "))
            mask_type = "original" if mask_type == 1 else "eroded"
        
        experiment_name = input("\nEnter experiment name: ")
        run_name = input("Enter model name: ")
        
        return {
            "model_type": "roads" if model_type == 1 else "buildings",
            "architecture": "unet" if architecture == 1 else "cnn",
            "mask_type": mask_type,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "use_reduced_dataset": use_reduced_dataset
        }

    def load_model_and_data(self, config):
        model_path = f"training_models.{config['model_type']}.model_factory"
        data_loader_path = f"training_models.{config['model_type']}.data_loader"
        mask_loader_path = f"training_models.{config['model_type']}.mask_loader"
        
        model_factory = __import__(model_path, fromlist=["ModelFactory"])
        data_loader = __import__(data_loader_path, fromlist=["DataLoader"])
        mask_loader = __import__(mask_loader_path, fromlist=["MaskLoader"])
        
        model = model_factory.ModelFactory.create_model(
            architecture=config["architecture"],
            mask_type=config["mask_type"]
        )
        
        data_loader = data_loader.DataLoader()
        
        return model, data_loader

    def train(self):
        config = self.get_user_input()
        
        self.logger.info("Starting training process...")
        if config["model_type"] == "buildings" and config["architecture"] == "cnn":
            self.logger.info(f"Using reduced dataset for buildings (CNN): {REDUCED_DATASET_SIZE['train']} train / {REDUCED_DATASET_SIZE['val']} val")
            self.batch_size = 4
            train_data = None
            val_data = None
        elif config["model_type"] == "buildings" and config["use_reduced_dataset"]:
            self.logger.info(f"Using reduced dataset for buildings: {REDUCED_DATASET_SIZE['train']} train / {REDUCED_DATASET_SIZE['val']} val")
            self.epochs = 20
            train_data = None
            val_data = None
        else:
            train_data = None
            val_data = None
        
        mlflow.tensorflow.autolog(disable=True)
        self.mlflow_manager = MLflowManager(config["experiment_name"], "Training run")
        self.mlflow_manager.start_run(run_name=config["run_name"])
        
        model, data_loader = self.load_model_and_data(config)

        if config["model_type"] == "buildings" and config["use_reduced_dataset"]:
            train_data = data_loader.load("train", limit_samples=True, mask_type=config["mask_type"])
            val_data = data_loader.load("val", limit_samples=True, mask_type=config["mask_type"])
        else:
            train_data = data_loader.load("train", mask_type=config["mask_type"])
            val_data = data_loader.load("val", mask_type=config["mask_type"])
        
        metrics = ["accuracy"]
        if config["model_type"] == "roads":
            metrics.extend([roads_dice_coefficient, roads_iou_score])
        else:
            metrics.extend([buildings_dice_coefficient, buildings_iou_score])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=metrics
        )
        
        self.mlflow_manager.log_params({
            "model_type": config["model_type"],
            "architecture": config["architecture"],
            "mask_type": config["mask_type"],
            "input_shape": self.input_shape,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "use_reduced_dataset": config["use_reduced_dataset"]
        })
        
        steps_per_epoch = max(1, train_data.cardinality().numpy() // self.batch_size)
        validation_steps = max(1, val_data.cardinality().numpy() // self.batch_size)
        
        callbacks = [
            self.tensorboard_manager.get_callback(experiment_name=config["experiment_name"]),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=5,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                verbose=1,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_dir, config["model_type"], f"best_{config['run_name']}.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            ),
            EpochLogger(log_dir=self.log_dir, run_name=config["run_name"])
        ]
        
        history = model.fit(
            train_data.repeat(),
            validation_data=val_data,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        final_metrics = {metric: history.history[metric][-1] for metric in history.history.keys()}
        
        output_path = os.path.join(self.output_dir, config["model_type"], f"{config['run_name']}.keras")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save(output_path)
        
        input_example = np.random.random((1, *self.input_shape)).astype(np.float32)
        self.mlflow_manager.log_model(model, model_name=config["model_type"], input_example=input_example)
        self.mlflow_manager.end_run()
        
        self.logger.info("Training process completed.")

if __name__ == "__main__":
    logger = ColorLogger("Main").get_logger()
    try:
        trainer = ModelTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred: {e}")