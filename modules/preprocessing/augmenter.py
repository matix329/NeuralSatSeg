import numpy as np

class Augmenter:
    def augment(self, image, mask):
        augmented = [(image, mask)]
        for angle in [90, 180, 270]:
            rotated_image = np.rot90(image, k=angle // 90, axes=(1, 2))
            rotated_mask = np.rot90(mask, k=angle // 90)
            augmented.append((rotated_image, rotated_mask))

        flipped_image = np.flip(image, axis=2)
        flipped_mask = np.flip(mask, axis=1)
        augmented.append((flipped_image, flipped_mask))
        return augmented