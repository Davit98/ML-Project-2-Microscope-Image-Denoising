import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="n2n_unet",
                        help="model architecture ('n2n_unet' or'srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="folder to save resulting images")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    model = get_model(args.model)
    model.load_weights(weight_file)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape

        out_image = np.zeros((h, w * 2, 3), dtype=np.uint8)

        pred = model.predict(np.expand_dims(image, 0))
        denoised_image = get_image(pred[0])

        out_image[:, :w] = image
        out_image[:, w: w * 2] = denoised_image


        cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)


if __name__ == '__main__':
    main()




