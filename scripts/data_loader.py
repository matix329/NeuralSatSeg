import tensorflow as tf

class DataLoader:
    def __init__(self, batch_size, image_size, num_classes):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes

    def load(self, image_dir, mask_dir=None):
        image_paths = tf.data.Dataset.list_files(f"{image_dir}/*.png", shuffle=True)

        if mask_dir:
            mask_paths = tf.data.Dataset.list_files(f"{mask_dir}/*.png", shuffle=True)
            dataset = tf.data.Dataset.zip((image_paths, mask_paths))

            def process(image_path, mask_path):
                image = self.load_image(image_path)
                mask = self.load_mask(mask_path)
                return image, mask

            dataset = dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            def process(image_path):
                image = self.load_image(image_path)
                return image

            dataset = image_paths.map(process, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, self.image_size) / 255.0
        image = tf.ensure_shape(image, [self.image_size[0], self.image_size[1], 3])
        return image

    def load_mask(self, mask_path):
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, self.image_size)
        mask = tf.ensure_shape(mask, [self.image_size[0], self.image_size[1], 1])
        mask = tf.math.round(mask)
        return mask