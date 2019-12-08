from keras.models import Model
from keras.layers import Input, Add, PReLU, LeakyReLU, Conv2DTranspose, concatenate, Concatenate, MaxPooling2D, \
    UpSampling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf


class L0Loss:
    def __init__(self):
        self.gamma = K.variable(2.)

    def __call__(self):
        def calc_loss(y_true, y_pred):
            loss = K.pow(K.abs(y_true - y_pred) + 1e-8, self.gamma)
            return loss
        return calc_loss


class UpdateAnnealingParameter(Callback):
    def __init__(self, gamma, nb_epochs, verbose=0):
        super(UpdateAnnealingParameter, self).__init__()
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        K.set_value(self.gamma, new_gamma)

        if self.verbose > 0:
            print('\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s.' % (epoch + 1, new_gamma))


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def get_model(model_name="n2n_unet", grayscale=False):
    input_channel_num = 1 if grayscale else 3
    output_channel_num = 1 if grayscale else 3

    if model_name == "srresnet":
        return get_srresnet_model(input_channel_num, output_channel_num)
    elif model_name == "unet":
        return get_unet_model(input_channel_num, output_channel_num)
    elif model_name == "n2n_unet":
        return get_n2n_unet(input_channel_num, output_channel_num)
    else:
        raise ValueError("model_name should be 'n2n_unet' or 'srresnet'or 'unet'")


# SRResNet
def get_srresnet_model(input_channel_num=3, feature_dim=64, resunit_num=16):
    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])

        return m

    inputs = Input(shape=(None, None, input_channel_num))
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])
    x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)

    return model


# UNet: code from https://github.com/pietz/unet-keras
def get_unet_model(input_channel_num=3, output_channel_num=3, start_ch=64, depth=4, inc_rate=2., activation='relu',
                   dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    def _conv_block(m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, activation=acti, padding='same')(m)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, activation=acti, padding='same')(n)
        n = BatchNormalization()(n) if bn else n

        return Concatenate()([m, n]) if res else n

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = _conv_block(m, dim, acti, bn, res)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same')(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            n = Concatenate()([n, m])
            m = _conv_block(n, dim, acti, bn, res)
        else:
            m = _conv_block(m, dim, acti, bn, res, do)

        return m

    i = Input(shape=(None, None, input_channel_num))
    o = _level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(output_channel_num, 1)(o)
    model = Model(inputs=i, outputs=o)

    return model


def get_n2n_unet(input_channel_num, output_channel_num):
    def conv_with_leaky_relu(filters, kernel_size, padding, alpha, input):
        conv = Conv2D(filters, kernel_size, padding=padding)(input)
        return LeakyReLU(alpha=alpha)(conv)

    def deconv_with_leaky_relu(filters, kernel_size, padding, alpha, input):
        dec_conv = Conv2D(filters, kernel_size, padding=padding)(input)
        return LeakyReLU(alpha=alpha)(dec_conv)

    inputs = Input((None, None, input_channel_num))

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

    dec_conv1c = Conv2D(output_channel_num, 3, activation='linear', padding='same')(dec_conv1b)

    model = Model(inputs=inputs, outputs=dec_conv1c)

    return model


def main():
    model = get_model("n2n_unet")
    model.summary()


if __name__ == '__main__':
    main()
