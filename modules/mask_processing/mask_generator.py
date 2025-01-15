from rasterio.features import rasterize

class MaskGenerator:
    def __init__(self, image_shape, transform):
        self.image_shape = image_shape
        self.transform = transform

    def generate_mask(self, geojson_data):
        geometries = [(feature.geometry, 1) for feature in geojson_data.itertuples()]
        mask = rasterize(
            geometries,
            out_shape=self.image_shape,
            transform=self.transform,
            fill=0,
            dtype="uint8"
        )
        return mask