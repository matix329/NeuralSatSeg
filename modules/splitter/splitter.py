import random
from typing import List, Tuple, Union
import numpy as np


class Splitter:
    def __init__(self, data: List[Tuple[str, Union[np.ndarray, str], str]], test_size: float = 0.2, shuffle: bool = True,seed: int = None):
        self.data = data
        self.test_size = test_size
        self.shuffle = shuffle
        self.seed = seed

    def split(self) -> Tuple[List[Tuple[str, np.ndarray, str]], List[Tuple[str, np.ndarray, str]]]:
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.data)

        split_idx = int(len(self.data) * (1 - self.test_size))
        train_data = self.data[:split_idx]
        val_data = self.data[split_idx:]

        return train_data, val_data