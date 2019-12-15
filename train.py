import argparse
import numpy as np
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from model import get_model, PSNR, L0Loss, UpdateAnnealingParameter
from generator import TrainImageGenerator, ValGenerator
import time


class Schedule:
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
    parser.add_argument("--source_image_dir", type=str, required=True,
                        help="source image dir")
    parser.add_argument("--target_image_dir", type=str, required=True,
                        help="target image dir")
    parser.add_argument("--source_val_dir", type=str, required=True,
                        help="validation set source image dir")
    parser.add_argument("--target_val_dir", type=str, required=True,
                        help="validation set target image dir")
    parser.add_argument("--image_size", type=int, default=512,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="weights",
                        help="weight dir")
    parser.add_argument("--model", type=str, default="n2n_unet",
                        help="model architecture, possible options 'n2n_unet'")
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # Files and paths
    source_image_dir = args.source_image_dir
    target_image_dir = args.target_image_dir
    source_val_dir = args.source_val_dir
    target_val_dir = args.target_val_dir
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Model parameters
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    loss_type = args.loss

    # Build model
    model = get_model(args.model)
    if args.weight is not None:
        model.load_weights(args.weight)
    opt = Adam(lr=lr)

    callbacks = []  # TODO move on some lines below

    # if loss_type == "l0": # TODO delete
    #     l0 = L0Loss()
    #     callbacks.append(UpdateAnnealingParameter(l0.gamma, nb_epochs, verbose=1))
    #     loss_type = l0()

    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])

    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}.hdf5",
                                     monitor="val_loss",
                                     verbose=1,
                                     period=10)
                     )

    # Add tensorboard logs
    NAME = "Noise2Noise-Biology-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
    callbacks.append(tensorboard)

    # Form train and validation sets
    train_generator = TrainImageGenerator(source_image_dir,
                                          target_image_dir,
                                          batch_size=batch_size, image_size=image_size)
    val_generator = ValGenerator(source_val_dir,
                                 target_val_dir)

    # Train
    hist = model.fit_generator(generator=train_generator,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
