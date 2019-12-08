from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence


class TrainImageGenerator(Sequence):
    def __init__(self, train_dir, batch_size=4, image_size=512):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.source_image_paths = [p for p in sorted(Path(train_dir + "/src").glob("**/*")) if
                                   p.suffix.lower() in image_suffixes]
        self.target_image_paths = [p for p in sorted(Path(train_dir + "/trg").glob("**/*")) if
                                   p.suffix.lower() in image_suffixes]

        if len(self.source_image_paths) != len(self.target_image_paths):
            raise ValueError("The number of source images is not equal to that of target images")

        self.image_num = len(self.source_image_paths)

        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)

        sample_id = 0

        for i in range(idx * batch_size, (idx + 1) * batch_size):
            x[sample_id] = cv2.imread(str(self.source_image_paths[i]))
            y[sample_id] = cv2.imread(str(self.target_image_paths[i]))

            sample_id += 1

        return x, y


class TestGenerator(Sequence):
    def __init__(self, test_dir):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")

        image_path_source = [p for p in sorted(Path(test_dir + "src").glob("**/*"))
                             if p.suffix.lower() in image_suffixes]
        image_path_target = [p for p in sorted(Path(test_dir + "trg").glob("**/*"))
                             if p.suffix.lower() in image_suffixes]

        self.image_num = len(image_path_source)
        self.data = []

        for i in range(self.image_num):
            x = cv2.imread(str(image_path_source[i]))
            y = cv2.imread(str(image_path_target[i]))

            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)
            self.data.append([x, y])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
