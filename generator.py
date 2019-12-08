from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence


class TrainImageGenerator(Sequence):
    def __init__(self, source_image_dir, target_image_dir, batch_size=32, image_size=512):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.source_image_paths = [p for p in sorted(Path(source_image_dir).glob("**/*")) if p.suffix.lower() in image_suffixes]
        self.target_image_paths = [p for p in sorted(Path(target_image_dir).glob("**/*")) if p.suffix.lower() in image_suffixes]

        if len(self.source_image_paths)!=len(self.target_image_paths): 
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

        for k in range(idx*batch_size,(idx+1)*batch_size):

            source_image_path = self.source_image_paths[k]
            source_image = cv2.imread(str(source_image_path))
            h, w, _ = source_image.shape

            target_image_path = self.target_image_paths[k]
            target_image = cv2.imread(str(target_image_path))

            if h >= image_size and w >= image_size:
                h, w, _ = source_image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                x[sample_id] = source_image[i:i + image_size, j:j + image_size]
                y[sample_id] = target_image[i:i + image_size, j:j + image_size]
                
                sample_id += 1

                if sample_id == batch_size:
                    return x, y


class ValGenerator(Sequence):
    def __init__(self, val_source_dir, val_target_dir):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")

        image_path_source = [p for p in sorted(Path(val_source_dir).glob("**/*")) if p.suffix.lower() in image_suffixes]
        image_path_target = [p for p in sorted(Path(val_target_dir).glob("**/*")) if p.suffix.lower() in image_suffixes]

        self.image_num = len(image_path_source)
        self.data = []

        for i in range(self.image_num):
            x = cv2.imread(str(image_path_source[i]))
            y = cv2.imread(str(image_path_target[i]))

            x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
            y = y.reshape(1,y.shape[0],y.shape[1],y.shape[2])
            self.data.append([x,y])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
