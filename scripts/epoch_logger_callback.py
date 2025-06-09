import os
import tensorflow as tf

class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, run_name):
        super().__init__()
        self.log_file_path = os.path.join(log_dir, f"{run_name}_logs.txt")
        self.run_name = run_name
        self.log_dir = log_dir
        self.ensure_log_directory_exists()
        with open(self.log_file_path, 'a') as f:
            pass

    def ensure_log_directory_exists(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_line = f"Epoch {epoch + 1}: "
        
        for key, value in logs.items():
            if isinstance(value, (float, int)):
                log_line += f"{key}={value:.4f} "
            else:
                log_line += f"{key}={value} "

        lr = 'N/A'
        if hasattr(self.model.optimizer, 'lr'):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        log_line += f"learning_rate={lr:.6f}" if isinstance(lr, (int, float)) else f"learning_rate={lr}"

        with open(self.log_file_path, 'a') as f:
            f.write(log_line + '\n')

        print(f"INFO: Epoch {epoch + 1} results logged to {self.log_file_path}") 