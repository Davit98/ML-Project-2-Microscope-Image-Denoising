from keras.models import Model
from keras.layers import Input, LeakyReLU, concatenate, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras import backend as K
import tensorflow as tf


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    """
    Compute peak signal-to-noise ratio
    :param y_true: Ground truth image
    :param y_pred: Denoised image
    :return:
    """
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def get_model(model_name="n2n_unet", image_size=512):
    if model_name == "n2n_unet":
        return get_n2n_unet(image_size=image_size)
    else:
        raise ValueError("model_name should be 'n2n_unet'")


def get_n2n_unet(input_channels_num=1, output_channels_num=1, image_size=512):
    """
    U-net architecture from noise2noise paper
    :param input_channels_num: Number of channels of input images ('1' for grayscale, '3' for rgb)
    :param output_channels_num: Number of channels of output images ('1' for grayscale, '3' for rgb)
    :param image_size: Image's size
    :return: U-net model
    """
    def conv_with_leaky_relu(filters, kernel_size, padding, alpha, input):
        conv = Conv2D(filters, kernel_size, padding=padding, kernel_initializer='he_normal')(input)
        return LeakyReLU(alpha=alpha)(conv)

    def deconv_with_leaky_relu(filters, kernel_size, padding, alpha, input):
        dec_conv = Conv2D(filters, kernel_size, padding=padding, kernel_initializer='he_normal')(input)
        return LeakyReLU(alpha=alpha)(dec_conv)

    inputs = Input((image_size, image_size, input_channels_num))

    enc_conv0 = conv_with_leaky_relu(filters=48, kernel_size=3, padding='same', alpha=0.1, input=inputs)
    enc_conv1 = conv_with_leaky_relu(filters=48, kernel_size=3, padding='same', alpha=0.1, input=enc_conv0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(enc_conv1)

    enc_conv2 = conv_with_leaky_relu(filters=48, kernel_size=3, padding='same', alpha=0.1, input=pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(enc_conv2)

    enc_conv3 = conv_with_leaky_relu(filters=48, kernel_size=3, padding='same', alpha=0.1, input=pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(enc_conv3)

    enc_conv4 = conv_with_leaky_relu(filters=48, kernel_size=3, padding='same', alpha=0.1, input=pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(enc_conv4)

    enc_conv5 = conv_with_leaky_relu(filters=48, kernel_size=3, padding='same', alpha=0.1, input=pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(enc_conv5)

    enc_conv6 = conv_with_leaky_relu(filters=48, kernel_size=3, padding='same', alpha=0.1, input=pool5)

    upsample5 = UpSampling2D(size=(2, 2))(enc_conv6)

    concat5 = concatenate([upsample5, pool4])

    dec_conv5a = deconv_with_leaky_relu(filters=96, kernel_size=3, padding='same', alpha=0.1, input=concat5)
    dec_conv5b = deconv_with_leaky_relu(filters=96, kernel_size=3, padding='same', alpha=0.1, input=dec_conv5a)

    upsample4 = UpSampling2D(size=(2, 2))(dec_conv5b)

    concat4 = concatenate([upsample4, pool3])

    dec_conv4a = deconv_with_leaky_relu(filters=96, kernel_size=3, padding='same', alpha=0.1, input=concat4)
    dec_conv4b = deconv_with_leaky_relu(filters=96, kernel_size=3, padding='same', alpha=0.1, input=dec_conv4a)

    upsample3 = UpSampling2D(size=(2, 2))(dec_conv4b)

    concat3 = concatenate([upsample3, pool2])

    dec_conv3a = deconv_with_leaky_relu(filters=96, kernel_size=3, padding='same', alpha=0.1, input=concat3)
    dec_conv3b = deconv_with_leaky_relu(filters=96, kernel_size=3, padding='same', alpha=0.1, input=dec_conv3a)

    upsample2 = UpSampling2D(size=(2, 2))(dec_conv3b)

    concat2 = concatenate([upsample2, pool1])

    dec_conv2a = deconv_with_leaky_relu(filters=96, kernel_size=3, padding='same', alpha=0.1, input=concat2)
    dec_conv2b = deconv_with_leaky_relu(filters=96, kernel_size=3, padding='same', alpha=0.1, input=dec_conv2a)

    upsample1 = UpSampling2D(size=(2, 2))(dec_conv2b)

    concat1 = concatenate([upsample1, inputs])

    dec_conv1a = deconv_with_leaky_relu(filters=64, kernel_size=3, padding='same', alpha=0.1, input=concat1)
    dec_conv1b = deconv_with_leaky_relu(filters=32, kernel_size=3, padding='same', alpha=0.1, input=dec_conv1a)

    dec_conv1c = Conv2D(output_channels_num, 3, activation='linear', padding='same')(dec_conv1b)

    model = Model(inputs=inputs, outputs=dec_conv1c)

    return model


def main():
    model = get_model("n2n_unet")
    model.summary()


if __name__ == '__main__':
    main()
