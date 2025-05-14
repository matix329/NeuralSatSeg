import tensorflow as tf
from config import MASK_PATHS

class DataLoader:
    def __init__(self, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size

    def load(self, data_dir):
        building_images = tf.data.Dataset.list_files(f"{data_dir}/buildings/images/*.png", shuffle=True)
        building_masks = tf.data.Dataset.list_files(f"{data_dir}/buildings/{MASK_PATHS['buildings']}/*.png", shuffle=False)
        ds_buildings = tf.data.Dataset.zip((building_images, building_masks))

        def process_building(image_path, mask_path):
            image = self.load_image(image_path)
            building_mask = self.load_mask(mask_path)
            road_mask = tf.zeros_like(building_mask)
            binary_map = tf.zeros_like(image[..., :1])
            return (image, binary_map), {'head_buildings': building_mask, 'head_roads': road_mask}

        ds_buildings = ds_buildings.map(process_building, num_parallel_calls=tf.data.AUTOTUNE)

        road_images = tf.data.Dataset.list_files(f"{data_dir}/roads/images/*.png", shuffle=True)
        road_masks = tf.data.Dataset.list_files(f"{data_dir}/roads/{MASK_PATHS['roads']}/*.png", shuffle=False)
        ds_roads = tf.data.Dataset.zip((road_images, road_masks))

        def process_road(image_path, mask_path):
            image = self.load_image(image_path)
            road_mask = self.load_mask(mask_path)
            building_mask = tf.zeros_like(road_mask)
            binary_map = tf.zeros_like(image[..., :1])
            return (image, binary_map), {'head_buildings': building_mask, 'head_roads': road_mask}

        ds_roads = ds_roads.map(process_road, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = ds_buildings.concatenate(ds_roads)
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
        mask = tf.cast(mask, tf.float32) / 255.0
        return mask