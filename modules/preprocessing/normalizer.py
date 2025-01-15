import numpy as np

class DataNormalizer:
    @staticmethod
    def normalize(image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))