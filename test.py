import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="n2n_unet",
                        help="model architecture, possible options 'n2n_unet'")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="folder to save the resulting images")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    """
    Test model
    :return:
    """
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(Path(image_dir).glob("*.*"))
    image_shape = np.load(str(image_paths[0])).shape
    assert(image_shape[0] == image_shape[1])
    image_size = image_shape[0]

    model = get_model(args.model, image_size=image_size)
    model.load_weights(weight_file)

    for image_path in image_paths:
        original_image = np.load(str(image_path)).reshape([image_size, image_size, 1])

        pred = model.predict(np.expand_dims(original_image, 0))
        denoised_image = pred[0]

        # Use this to save denoised images as numpy arrays
        # np.save(str(output_dir.joinpath(image_path.name))[:-4] + "_source", image)
        # np.save(str(output_dir.joinpath(image_path.name))[:-4] + "_denoised", denoised_image)

        original_image = original_image.reshape((image_size, image_size))
        denoised_image = denoised_image.reshape((image_size, image_size))
        plt.imsave(str(output_dir.joinpath(image_path.name))[:-4] + "_original.jpg", original_image, cmap='gray')
        plt.imsave(str(output_dir.joinpath(image_path.name))[:-4] + "_denoised.jpg", denoised_image, cmap='gray')


if __name__ == '__main__':
    main()
