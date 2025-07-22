import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

MASK_OPTIONS = ["binary", "graph"]

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/mlflow_artifacts/models/roads")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs") 