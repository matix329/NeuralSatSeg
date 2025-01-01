import tensorflow as tf

class DataLoader:
    def __init__(self, batch_size, image_size, num_classes):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes

    def load(self, image_dir, mask_dir):
        image_paths = tf.data.Dataset.list_files(image_dir + "/*/*.jpg", shuffle=True)
        mask_paths = tf.data.Dataset.list_files(mask_dir + "/*/*.png", shuffle=True)

        dataset = tf.data.Dataset.zip((image_paths, mask_paths))
        dataset = dataset.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    def process_path(self, image_path, mask_path):
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)
        return image, mask

    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size) / 255.0
        return image

    def load_mask(self, mask_path):
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, self.image_size, method="nearest")
        mask = tf.cast(mask, tf.int32)
        mask = tf.clip_by_value(mask, 0, self.num_classes - 1)
        return mask