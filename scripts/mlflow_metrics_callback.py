import tensorflow as tf

class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, mlflow_manager):
        super().__init__()
        self.mlflow_manager = mlflow_manager
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = {}
        
        for key, value in logs.items():
            if isinstance(value, (float, int)):
                metrics[key] = float(value)
        
        for key, value in metrics.items():
            self.mlflow_manager.log_metric(key, value, step=epoch) 