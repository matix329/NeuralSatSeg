import os
from tensorflow.keras.callbacks import TensorBoard

class TensorboardManager:
    def __init__(self, log_dir="../mlflow_artifacts/tensorboard_logs", histogram_freq=1, write_graph=True, write_images=False):
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images

    def get_callback(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        return TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=self.histogram_freq,
            write_graph=self.write_graph,
            write_images=self.write_images,
        )