import tensorflow as tf
import os

class DataLoader:
    def __init__(self, image_size=(64, 64), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size

    def load(self, image_dir, mask_dir):
        image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith((".jpg", ".png"))])
        mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith(".png")])

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(self._process_path).batch(self.batch_size)

        return dataset

    def _process_path(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size) / 255.0

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, self.image_size, method='nearest')
        mask = tf.cast(mask, tf.int32)

        return image, mask