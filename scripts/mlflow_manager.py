import mlflow.keras
from mlflow.models.signature import infer_signature

class MLflowManager:
    def __init__(self, experiment_name):
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def start_run(self, run_name=None):
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_epoch_metrics(self, history, epoch):
        metrics = {
            "train_accuracy": history.history["accuracy"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "train_loss": history.history["loss"][-1],
            "val_loss": history.history["val_loss"][-1],
        }
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_model(self, model, model_name, input_example):
        artifact_path = model_name
        signature = infer_signature(input_example, model.predict(input_example))

        mlflow.keras.log_model(
            model,
            artifact_path=artifact_path,
            input_example=input_example,
            signature=signature
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model(model_uri, model_name)

    def end_run(self):
        mlflow.end_run()