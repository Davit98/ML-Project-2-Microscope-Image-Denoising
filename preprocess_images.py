import argparse
import nd2reader
import matplotlib.pyplot as plt
import os
import numpy as np
from os import listdir
import shutil

from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from skimage.util import *
from skimage import exposure
import cv2


def read_data(reader, data_path):
    """
    Reads data from .nd2 files.
    :param reader: nd2 reader to use
    :param data_path: path to files to read
    :return: list of samples (content of file) and list of filenames.
    """
    samples = []
    filenames = []
    for filename in sorted(listdir(data_path)):
        if filename.endswith('.nd2'):
            samples.append(reader(data_path + "/" + filename))
            filenames.append(filename)
    return samples, filenames


def get_volume(sample, channel, frame):
    """
    Loads particular volume from sample.
    :param sample: sample from which load volume.
    :param channel: which channel to read.
    :param frame: which frame to read.
    :return: volume which is np.array containing images of all levels.
    """
    if channel in range(sample.sizes["c"]) and frame in range(sample.sizes["t"]):
        sample.iter_axes = 'z'
        sample.default_coords['c'] = channel
        sample.default_coords['t'] = frame

        volume = np.array([np.array(level) for level in sample])
        return volume


def align_images(img_1, img_2):
    """
    Aligns first image to second one using cross-correlation in Fourier space to find shift.
    :param img_1: first image.
    :param img_2: second image.
    :return: pair of images: (shifted version of image1 and image2.
    """
    shift, error, diffphase = register_translation(img_1, img_2, 100)

    shift = -1 * shift
    img_1_shifted = fourier_shift(np.fft.fftn(img_1), shift)
    img_1_shifted = np.fft.ifftn(img_1_shifted).real

    return img_1_shifted, img_2


def align_volumes(volume_1, volume_2):
    """
    Aligns volumes by aligning each pair of images from two volumes.
    Pair contains one image per volume with the same level.
    :param volume_1: first volume.
    :param volume_2: second volume.
    :return: pair of volumes: (shifted version of volume1 and volume2).
    """
    volume_1_shifted = []
    for img_1, img_2 in zip(volume_1, volume_2):
        img_1_shifted, img_2 = align_images(img_1, img_2)
        volume_1_shifted.append(img_1_shifted)
    return volume_1_shifted, volume_2


def augment_image(img):
    """
    Augments image by using flipping image across both axes.
    :param img: image to augment.
    :return: list of 4 images (first image is original).
    """
    return [img, np.fliplr(img), np.flipud(img), np.fliplr(np.flipud(img))]


def augment_volumes(volume_1, volume_2):
    """
    Augments pair of volumes by flipping each image across both axes.
    :param volume_1: first volume.
    :param volume_2: second volume.
    :return: pair of augmented volumes.
    """
    volume_1_augmented = []
    volume_2_augmented = []

    for img_1, img_2 in zip(volume_1, volume_2):
        volume_1_augmented.extend(augment_image(img_1))
        volume_2_augmented.extend(augment_image(img_2))
    return volume_1_augmented, volume_2_augmented


def is_overnoised(img, bright_pixel_percent=0.05):
    """
    Checks whether image is overnoised.
    Uses simple heuristic: overnoised images have big percentage of "bright" pixels.
    :param img: image to check.
    :param bright_pixel_percent: threshold (max percentage of "bright" pixels in not overnoised image
    :return: 'True' if image is overnoised, 'False' otherwise.
    """
    hist = exposure.histogram(img, nbins=2)[0]
    value = hist[1]  # Number of bright points. (The less value, the less noise)
    n_pixels = img.shape[0] * img.shape[1]
    return value > n_pixels * bright_pixel_percent


def clean_volumes(volume_1, volume_2):
    """
    Delete pair of images from both volumes if one of images from pair is overnoised.
    :param volume_1: first volume.
    :param volume_2: second volume.
    :return: pair of cleaned volumes.
    """
    volume_1_cleaned = []
    volume_2_cleaned = []

    for img_1, img_2 in zip(volume_1, volume_2):
        if is_overnoised(img_1) | is_overnoised(img_2):
            continue
        else:
            volume_1_cleaned.append(img_1)
            volume_2_cleaned.append(img_2)
    return volume_1_cleaned, volume_2_cleaned


def save_img_grayscale(img, filename):
    """
    Save image in grayscale format.
    :param img: pixel matrix of image. Must be grayscale.
    :param filename: filename for file to save.
    :return: 'True' if image is successfully saved.
    """
    np.save(filename, img)
    # plt.imsave(filename, img, cmap="gray")
    return True


def save_volume(volume, path):
    """
    Save images form volume in grayscale format.
    :param volume: volume to save.
    :param path: path where save volume in.
    :return: 'True' if volume is successfully saved.
    """
    for ind, image in enumerate(volume):
        # filename = f"{path}_image_{ind}.jpg"
        filename = f"{path}_image_{ind}"
        save_img_grayscale(image, filename)
    return True


def create_folders(dataset_path):
    """
    Create folders for dataset.
    :param dataset_path: dataset's path where the folders will be created.
    :return: 'True' if folders are successfully created.
    """
    os.mkdir(dataset_path)

    for channel in ["green", "red"]:
        os.mkdir(f"{dataset_path}/{channel}")
        for train_or_val in ["train", "val"]:
            os.mkdir(f"{dataset_path}/{channel}/{train_or_val}", )
            for src_or_trg in ["src", "trg"]:
                os.mkdir(f"{dataset_path}/{channel}/{train_or_val}/{src_or_trg}")

    # # TODO delete
    # dataset_folders = [dataset_path,
    #                    f"{dataset_path}/train",
    #                    f"{dataset_path}/train/src",
    #                    f"{dataset_path}/train/trg",
    #                    f"{dataset_path}/val",
    #                    f"{dataset_path}/val/src",
    #                    f"{dataset_path}/val/trg"]
    #
    # for folder in dataset_folders:
    #     os.mkdir(folder)

    return True


def generate_dataset(samples, param, dataset_path):
    """
    Generate dataset.
    :param samples: samples from which take images for dataset.
    :param param: dictionary which contains set of preprocessing steps to be done.
    Possible pre-processing steps: 'clean', 'align', 'augment'.
    :param dataset_path: path where save dataset in.
    :return: 'True' if dataset is successfully generated.
    """
    create_folders(dataset_path)

    for sample_ind, sample in enumerate(samples):
        print("Sample №{}".format(sample_ind))

        for channel_ind, channel_name in enumerate(["green", "red"]):
            for pair_ind, pair in enumerate([[0, 2], [1, 3]]):  # Frame pairs

                is_test = (pair_ind == 1 and sample_ind == len(samples) - 1)

                volume_1 = get_volume(sample, channel=channel_ind, frame=pair[0])
                volume_2 = get_volume(sample, channel=channel_ind, frame=pair[1])

                if param.get('align'):
                    volume_1, volume_2 = align_volumes(volume_1, volume_2)

                if not is_test:  # We don't need to do this steps for test set
                    if param.get('clean'):
                        volume_1, volume_2 = clean_volumes(volume_1, volume_2)

                    if param.get('augment'):
                        volume_1, volume_2 = augment_volumes(volume_1, volume_2)

                assert (len(volume_1) == len(volume_2))

                train_or_val = "val" if is_test else "train"
                path_src = f"{dataset_path}/{channel_name}/{train_or_val}/src/sample_{sample_ind}_pair{pair_ind}"
                path_trg = f"{dataset_path}/{channel_name}/{train_or_val}/trg/sample_{sample_ind}_pair{pair_ind}"

                # Use this to save both channels in one directory
                # path_src = f"{dataset_path}/{train_or_val}/src/sample_{sample_ind}_channel_{channel_name}_pair_{pair_ind}"
                # path_trg = f"{dataset_path}/{train_or_val}/trg/sample_{sample_ind}_channel_{channel_name}_pair_{pair_ind}"

                save_volume(volume_1, path_src)
                save_volume(volume_2, path_trg)

                print(f"Channel: {channel_name}, "
                      f"Frames: {pair}, "
                      f"Number of samples: {len(volume_1)}, "
                      f"{train_or_val} set")

    return True


def delete_dataset(dataset_folder):
    """
    Delete folder and all it's content.
    :param dataset_folder: folder to delete
    :return: 'True' if dataset is successfully deleted.
    """
    if os.path.isdir(dataset_folder):
        shutil.rmtree(dataset_folder, ignore_errors=True)


def get_args():
    parser = argparse.ArgumentParser(description="preprocess .nd2 images",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="samples files dir (.nd2 format)")
    parser.add_argument("--data_type", type=str, default="",
                        help="defines which type of images to use. Can be '30ms', '5ms'. "
                             "If arg is not provided all data is used")
    args = parser.parse_args()

    return args


def main():
    """
    Generates dataset with pre-processing steps done from sample files
    :return:
    """
    args = get_args()
    data_dir = args.data_dir
    data_type = args.data_type

    reader = nd2reader.ND2Reader
    samples, filenames = read_data(reader, data_dir)
    if data_type:  # Filter only needed samples
        file_indices = [ind for ind, filename in enumerate(filenames) if data_type in filename]
        samples = [samples[ind] for ind in file_indices]
    print(f"Found in total {len(samples)} files:")

    # Experiment params
    params = [
        {'clean': False, 'align': True, 'augment': False},
        {'clean': True, 'align': True, 'augment': False},
        {'clean': False, 'align': True, 'augment': True},
        {'clean': True, 'align': True, 'augment': True}
    ]

    for experiment_ind, param in enumerate(params):
        experiment_name = "_".join([key for key, el in param.items() if el])

        print(f"Generating the dataset for experiment {experiment_name}")
        dataset_path = f"{experiment_name}_data_{data_type}"

        delete_dataset(dataset_path)
        generate_dataset(samples, param, dataset_path)


if __name__ == '__main__':
    main()
