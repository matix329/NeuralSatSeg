import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

TRAINING_CONFIG = {
    "input_shape": (650, 650, 3),
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001
}

MASK_OPTIONS = ["binary", "graph"]

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/mlflow_artifacts/models/roads")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs") 