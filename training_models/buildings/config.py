import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

TRAINING_CONFIG = {
    "input_shape": (640, 640, 3),
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.0001
}

MASK_OPTIONS = ["original", "eroded"]

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/mlflow_artifacts/models/buildings")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

REDUCED_DATASET_SIZE = {
    "train": 600,
    "val": 150
} 