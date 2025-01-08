import time
import os
from tensorflow.keras.callbacks import TensorBoard


class TensorboardManager:
    def __init__(self, log_dir="../mlflow_artifacts/tensorboard_logs", histogram_freq=1, write_graph=True,
                 write_images=False):
        self.base_log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images

    def get_callback(self, experiment_name):
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.base_log_dir, experiment_name, timestamp)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        return TensorBoard(
            log_dir=log_dir,
            histogram_freq=self.histogram_freq,
            write_graph=self.write_graph,
            write_images=self.write_images,
        )