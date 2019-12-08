from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence


class TrainImageGenerator(Sequence):
    def __init__(self, source_image_dir, target_image_dir, batch_size=32, image_size=512):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.source_image_paths = [p for p in Path(source_image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.target_image_paths = [p for p in Path(target_image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]

        if len(self.source_image_paths)!=self.target_image_paths: 
            raise ValueError("The number of source images is not equal to that of target images")

        self.image_num = len(self.source_image_paths)

        # shuffle images
        prm = np.random.permutation(len(self.image_num))
        self.source_image_paths = [self.source_image_paths[elem] for elem in prm]
        self.target_image_paths = [self.target_image_paths[elem] for elem in prm]

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

        while True:

            source_image_path = self.source_image_paths[sample_id]
            source_image = cv2.imread(str(source_image_path))
            h, w, _ = source_image.shape

            target_image_path = self.target_image_paths[sample_id]
            target_image = cv2.imread(str(target_image_path))

            if h >= image_size and w >= image_size:
                h, w, _ = source_image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                clean_patch_s = source_image[i:i + image_size, j:j + image_size]
                clean_patch_t = target_image[i:i + image_size, j:j + image_size]
                
                x[sample_id] = self.source_noise_model(clean_patch_s)
                y[sample_id] = self.target_noise_model(clean_patch_t)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y


class ValGenerator(Sequence):
    def __init__(self, val_source_dir, val_target_dir):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")

        image_path_source = [p for p in Path(val_source_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        image_path_target = [p for p in Path(val_target_dir).glob("**/*") if p.suffix.lower() in image_suffixes]

        self.image_num = len(image_path_source)
        self.data = []

        # if self.image_num == 0:
        #     raise ValueError("image dir '{}' does not include any image".format(image_dir))

        # for image_path in image_paths:
        #     y = cv2.imread(str(image_path))
        #     h, w, _ = y.shape
        #     y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
        #     x = val_noise_model(y)
        #     self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

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
