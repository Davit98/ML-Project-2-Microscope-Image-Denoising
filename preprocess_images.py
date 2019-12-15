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
            samples.append(reader(data_path + filename))
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


def save_img_grayscale(img, sample_index, channel, z, frame, augment_ind, path):  # Join all these arguments in one
    """
    Save image in grayscale format
    :param img: pixel matrix of image. Must be grayscale.
    :param sample_index:
    :param channel:
    :param z:
    :param frame:
    :param augment_ind:
    :param path: path where save image to
    :return: 'True' if image is successfully saved.
    """
    # image_name = 'image_s' + str(sample_index) + '_c' + str(channel) + '_z' + str(z) + '_f' + str(frame) + '_' + str(
    #     augment_ind) + '.jpg'
    image_name = f"image_s{sample_index}_c{channel}_z{z}_f{frame}_{str(augment_ind)}.jpg"
    file_path = path + '/' + image_name
    #     imageio.imwrite(path,np.array(img,dtype=np.uint8))
    plt.imsave(file_path, img, cmap='gray')
    return True


def save_dataset(data_type, filenames, params, exp_name):  # TODO refactor
    """
    Saves dataset
    :param data_type: type of images, possible options: '30ms', '5ms'
    :param filenames:
    :param params:
    :param exp_name:
    :return:
    """
    possible_data_types = ['30ms', '5ms']
    if data_type not in possible_data_types:
        raise ValueError("data_type should be '30ms' or '5ms'")

    DATA_PATH = exp_name + '_data_' + data_type

    SOURCE_PATH = '/source'
    TARGET_PATH = '/target'

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
        os.mkdir(DATA_PATH + SOURCE_PATH)
        os.mkdir(DATA_PATH + TARGET_PATH)
        os.mkdir(DATA_PATH + SOURCE_PATH + '/channel0')
        os.mkdir(DATA_PATH + SOURCE_PATH + '/channel1')
        os.mkdir(DATA_PATH + TARGET_PATH + '/channel0')
        os.mkdir(DATA_PATH + TARGET_PATH + '/channel1')

    file_indices = [i for i, elem in enumerate(filenames) if data_type in elem]

    for c in range(2):
        path1 = os.getcwd() + '/' + DATA_PATH + SOURCE_PATH + '/channel' + str(c)
        path2 = os.getcwd() + '/' + DATA_PATH + TARGET_PATH + '/channel' + str(c)

        volumes = []
        for file_index in file_indices:

            is_test = file_index == file_indices[0]

            for i in range(4):
                volumes.append(get_volume(samples[file_index], channel=c, frame=i))

            pairs = [(0, 2), (1, 3)]

            if is_test:
                pairs = [(1, 3)]

                TEST_SET_PATH = DATA_PATH + '/test_for_channel' + str(c)
                os.mkdir(TEST_SET_PATH)

                for ind, elem in enumerate(volumes[0]):
                    save_img_grayscale(elem, file_index, c, ind, 0, '_', TEST_SET_PATH)

                for ind, elem in enumerate(volumes[2]):
                    save_img_grayscale(elem, file_index, c, ind, 2, '_', TEST_SET_PATH)

            if params['clean']:
                for p in pairs:
                    volumes[p[0]], volumes[p[1]] = clean_volumes(volumes[p[0]], volumes[p[1]])

            if params['align']:
                for p in pairs:
                    volumes[p[0]], volumes[p[1]] = align_volumes(volumes[p[0]], volumes[p[1]])

            if params['augment']:
                for p in pairs:
                    volumes[p[0]], volumes[p[1]] = augment_volumes(volumes[p[0]], volumes[p[1]])

            for i, each in enumerate(volumes):
                if i == 0 or i == 1:
                    path = path1
                else:
                    path = path2

                for ind, elem in enumerate(each):
                    save_img_grayscale(elem, file_index, c, ind // 4 + 1, i, ind % 4, path)


def delete_dataset(dataset_folder):
    """
    Delete folder and all it's content.
    :param dataset_folder: folder to delete
    :return: 'True' if dataset is successfully deleted.
    """
    if os.path.isdir(dataset_folder):
        shutil.rmtree(dataset_folder, ignore_errors=True)


def main():
    DATA_PATH = 'data/'

    reader = nd2reader.ND2Reader
    samples, filenames = read_data(reader, DATA_PATH)
    print(f"Found in total {len(filenames)} files:")
    print(*filenames, sep="\n")

    # Experiment params
    params = [{'clean': False, 'align': True, 'augment': False},
              {'clean': True, 'align': True, 'augment': False},
              {'clean': False, 'align': True, 'augment': True},
              {'clean': True, 'align': True, 'augment': True}]

    data_type = "5ms"
    for exp_ind, param in enumerate(params):
        print("Saving the dataset for experiment " + str(exp_ind + 1))
        exp_name = "_".join([key for key, el in param.items() if el])

        #     delete_dataset() # TODO
        save_dataset(data_type, filenames, param, exp_name)
