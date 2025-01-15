import mlflow.keras
from mlflow.models.signature import infer_signature
import os
import json

class MLflowManager:
    def __init__(self, experiment_name):
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_example_dir = os.path.join(base_dir, "../mlflow_artifacts/input_examples")
        os.makedirs(self.input_example_dir, exist_ok=True)

    def start_run(self, run_name=None):
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_epoch_metrics(self, history, epoch):
        metrics = {
            "train_accuracy": history.history["accuracy"][-1],
            "train_loss": history.history["loss"][-1],
        }

        if "val_accuracy" in history.history:
            metrics["val_accuracy"] = history.history["val_accuracy"][-1]
        if "val_loss" in history.history:
            metrics["val_loss"] = history.history["val_loss"][-1]

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_model(self, model, model_name, input_example):
        input_example_path = os.path.join(self.input_example_dir, f"{self.experiment_name}_input_example.json")

        with open(input_example_path, "w") as f:
            json.dump(input_example.tolist(), f)

        signature = infer_signature(input_example, model.predict(input_example))

        mlflow.keras.log_model(
            model,
            artifact_path=model_name,
            signature=signature
        )

    def end_run(self):
        mlflow.end_run()