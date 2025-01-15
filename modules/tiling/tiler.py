import numpy as np

class Tiler:
    def __init__(self, tile_size):
        self.tile_size = tile_size

    def tile(self, image, mask):
        h, w = image.shape[-2], image.shape[-1]
        tiles = []
        for i in range(0, h, self.tile_size):
            for j in range(0, w, self.tile_size):
                img_tile = image[..., i:i+self.tile_size, j:j+self.tile_size]
                mask_tile = mask[i:i+self.tile_size, j:j+self.tile_size]
                if img_tile.shape[-2:] != (self.tile_size, self.tile_size):
                    padded_img_tile = np.zeros((image.shape[0], self.tile_size, self.tile_size), dtype=image.dtype)
                    padded_mask_tile = np.zeros((self.tile_size, self.tile_size), dtype=mask.dtype)
                    padded_img_tile[..., :img_tile.shape[-2], :img_tile.shape[-1]] = img_tile
                    padded_mask_tile[:mask_tile.shape[-2], :mask_tile.shape[-1]] = mask_tile
                    tiles.append((padded_img_tile, padded_mask_tile))
                else:
                    tiles.append((img_tile, mask_tile))
        return tiles