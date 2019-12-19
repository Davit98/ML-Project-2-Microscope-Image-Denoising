from pathlib import Path
import numpy as np
from keras.utils import Sequence

FILE_FORMATS = ".npy"


class TrainImageGenerator(Sequence):
    """
    Generator for train set of images
    """
    def __init__(self, train_image_dir, batch_size=4, image_size=512):
        self.source_image_paths = [p for p in sorted(Path(train_image_dir + "/src").glob("**/*")) if
                                   p.suffix.lower() in FILE_FORMATS]
        self.target_image_paths = [p for p in sorted(Path(train_image_dir + "/trg").glob("**/*")) if
                                   p.suffix.lower() in FILE_FORMATS]

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
            source_image = np.load(str(source_image_path)).reshape([image_size, image_size, 1])

            target_image_path = self.target_image_paths[k]
            target_image = np.load(str(target_image_path)).reshape([image_size, image_size, 1])

            x[k - idx * batch_size] = source_image
            y[k - idx * batch_size] = target_image

        return x, y


class ValImageGenerator(Sequence):
    """
    Generator for validation set of images
    """
    def __init__(self, val_image_dir, image_size):
        image_path_source = [p for p in sorted(Path(val_image_dir + "/src").glob("**/*")) if p.suffix.lower() in FILE_FORMATS]
        image_path_target = [p for p in sorted(Path(val_image_dir + "/trg").glob("**/*")) if p.suffix.lower() in FILE_FORMATS]

        self.image_num = len(image_path_source)
        self.data = []

        for i in range(self.image_num):
            x = np.load(str(image_path_source[i])).reshape([1, image_size, image_size, 1])
            y = np.load(str(image_path_target[i])).reshape([1, image_size, image_size, 1])

            self.data.append([x, y])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
