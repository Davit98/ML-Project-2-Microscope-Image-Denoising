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
        # image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image = np.load(str(image_path)).reshape([image_size, image_size, 1])

        out_image = np.zeros((image_size, image_size * 2, 1), dtype=np.uint8)

        pred = model.predict(np.expand_dims(image, 0))
        denoised_image = pred[0]

        np.save(str(output_dir.joinpath(image_path.name))[:-4] + "_denoised", denoised_image)
        np.save(str(output_dir.joinpath(image_path.name))[:-4] + "_source", image)

        # image = get_image(image)
        # denoised_image = get_image(denoised_image)

        # np.save(str(output_dir.joinpath(image_path.name))[:-4] + "_denoised_clipped", denoised_image)
        # np.save(str(output_dir.joinpath(image_path.name))[:-4] + "_source_clipped", image)

        # print(image.min(), image.max(), image.mean())
        # print(denoised_image.min(), denoised_image.max(), denoised_image.mean())

        # Place denoised image next to original one
        out_image[:, :image_size] = image
        out_image[:, image_size:image_size * 2] = denoised_image

        out_image = out_image.reshape([image_size, image_size * 2])

        # cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)
        # cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)  # TODO change to .npy

        # plt.imsave(str(output_dir.joinpath(image_path.name))[:-4] + ".jpg", out_image, cmap='gray')


if __name__ == '__main__':
    main()
