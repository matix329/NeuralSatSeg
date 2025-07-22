import tensorflow as tf
import numpy as np

class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, mlflow_manager):
        super().__init__()
        self.mlflow_manager = mlflow_manager
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = {}
        
        for key, value in logs.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 0 and isinstance(value[0], (float, int)):
                arr = np.array(value)
                metrics[f"{key}_mean"] = float(arr.mean())
                metrics[f"{key}_std"] = float(arr.std())
                metrics[f"{key}_min"] = float(arr.min())
                metrics[f"{key}_max"] = float(arr.max())
            elif isinstance(value, (float, int)):
                metrics[key] = float(value)
        
        for key, value in metrics.items():
            self.mlflow_manager.log_metric(key, value, step=epoch) 