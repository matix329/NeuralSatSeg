class MaskValidator:
    @staticmethod
    def validate_masks(images, masks):
        for (img_name, img), (mask_name, mask) in zip(images, masks):
            assert img.shape[1:] == mask.shape, f"Size error: {img_name} vs {mask_name}"