import argparse
import numpy as np
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from model import get_model, PSNR
from generator import TrainImageGenerator, ValImageGenerator
from time import strftime, gmtime


class Schedule:
    """
    Scheduler for decreasing learning rate over epochs
    """
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_image_dir", type=str, required=True,
                        help="train set image dir")
    parser.add_argument("--val_image_dir", type=str, required=True,
                        help="validation set image dir")
    parser.add_argument("--image_size", type=int, default=512,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="weights",
                        help="weight dir")
    parser.add_argument("--model", type=str, default="n2n_unet",
                        help="model architecture, possible options 'n2n_unet'")
    parser.add_argument("--desc", type=str, default="",
                        help="Experiment description. It will be used to name trained weights")
    args = parser.parse_args()

    return args


def main():
    """
    Training model
    :return:
    """
    args = get_args()
    experiment_desc = args.desc

    # Files and paths
    train_image_dir = args.train_image_dir
    val_image_dir = args.val_image_dir
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Model parameters
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    loss_type = args.loss

    # Build model
    model = get_model(args.model, image_size)
    if args.weight is not None:
        model.load_weights(args.weight)
    opt = Adam(lr=lr)

    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])

    callbacks = []
    # Decrease learning rate over epochs
    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))

    # Saving weights per 10 epochs
    weights_filename = str(output_path) + "/weights_" + experiment_desc + ".{epoch:03d}-{val_loss:.3f}.hdf5"
    callbacks.append(ModelCheckpoint(weights_filename, monitor="val_loss", verbose=1, period=10))

    # Add tensorboard logs
    logs_filename = "logs {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    tensorboard = TensorBoard(update_freq="epoch",
                              write_images=True,
                              log_dir='logs/{}'.format(logs_filename)
                              )
    callbacks.append(tensorboard)

    # Form train and validation sets
    train_generator = TrainImageGenerator(train_image_dir, batch_size=batch_size, image_size=image_size)
    val_generator = ValImageGenerator(val_image_dir, image_size=image_size)

    # Train
    hist = model.fit_generator(generator=train_generator,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
