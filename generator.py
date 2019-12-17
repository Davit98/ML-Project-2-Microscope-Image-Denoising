from pathlib import Path
import numpy as np
import cv2
from keras.utils import Sequence

IMAGE_FORMATS = (".jpeg", ".jpg", ".png", ".bmp")


class TrainImageGenerator(Sequence):
    """
    Generator for train set of images
    """
    def __init__(self, train_image_dir, batch_size=4, image_size=512):
        self.source_image_paths = [p for p in sorted(Path(train_image_dir + "/src").glob("**/*")) if
                                   p.suffix.lower() in IMAGE_FORMATS]
        self.target_image_paths = [p for p in sorted(Path(train_image_dir + "/trg").glob("**/*")) if
                                   p.suffix.lower() in IMAGE_FORMATS]

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
        x = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint8)

        for k in range(idx * batch_size, (idx + 1) * batch_size):
            source_image_path = self.source_image_paths[k]
            source_image = cv2.imread(str(source_image_path), cv2.IMREAD_GRAYSCALE).reshape(image_size, image_size, 1)

            target_image_path = self.target_image_paths[k]
            target_image = cv2.imread(str(target_image_path), cv2.IMREAD_GRAYSCALE).reshape(image_size, image_size, 1)

            x[k - idx * batch_size] = source_image
            y[k - idx * batch_size] = target_image

        return x, y


class ValImageGenerator(Sequence):
    """
    Generator for validation set of images
    """
    def __init__(self, val_image_dir):
        image_path_source = [p for p in sorted(Path(val_image_dir + "/src").glob("**/*")) if p.suffix.lower() in IMAGE_FORMATS]
        image_path_target = [p for p in sorted(Path(val_image_dir + "/trg").glob("**/*")) if p.suffix.lower() in IMAGE_FORMATS]

        self.image_num = len(image_path_source)
        self.data = []

        for i in range(self.image_num):
            x = cv2.imread(str(image_path_source[i]), cv2.IMREAD_GRAYSCALE)
            y = cv2.imread(str(image_path_target[i]), cv2.IMREAD_GRAYSCALE)

            x = x.reshape(1, x.shape[0], x.shape[1], 1)
            y = y.reshape(1, y.shape[0], y.shape[1], 1)
            self.data.append([x, y])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
