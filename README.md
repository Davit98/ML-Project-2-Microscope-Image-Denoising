# Removing noise from microscope images without ground truth
Learning to denoise microscope images of warm brain neurones without clean targets.  

#### Prerequisites

* nd2reader 3.2.1
* Numpy 1.15.4
* Skimage 0.15.0 
* cv2 3.4.1
* Tensorflow 1.15.0
* Keras 2.2.4


#### How to run

To train: ```python3 train.py --train_image_dir <YOUR_TRAINING_DATASET> --val_image_dir <YOUR_VALIDATION_DATASET> --batch_size <BATCH_SIZE> --image_size <IMAGE_SIZE> --lr <LEARNING_RATE> --nb_epochs <NUMBER_OF_EPOCHS> --output_path <FOLDER_TO_SAVE_WEIGHTS>``` 

To test: ```python3...```


#### File descriptions

model.py - contains keras implementation of the Unet described in the Noise2Noise paper

preprocess_images.py - contains methods for reading, preprocessing and saving data as numpy arrays:
* read data from .nd2 files
* align, augment and clean the data
* save data as numpy array

train.py - trains the Unet model

test.py - tests by loading the learning


generator.py - ...
