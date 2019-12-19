# Removing Noise From Microscope Images Without Fround Truth
Learning to denoise microscope images of warm brain neurones without clean targets.  

#### Prerequisites

* nd2reader 3.2.1
* Numpy 1.15.4
* Skimage 0.15.0 
* cv2 3.4.1
* Tensorflow 1.15.0
* Keras 2.2.4

#### How to run

To train: ```python3 train.py --train_image_dir <YOUR_TRAINING_DATASET_DIRECTORY> --val_image_dir <YOUR_VALIDATION_DATASET_DIRECTORY> --batch_size <BATCH_SIZE> --image_size <IMAGE_SIZE> --lr <LEARNING_RATE> --nb_epochs <NUMBER_OF_EPOCHS> --output_path <FOLDER_TO_SAVE_WEIGHTS>``` 

To test: ```python3 test_model.py --weight_file <WEIGHTS> --image_dir <TEST_DATASET_DIRECTORY> --output_dir <FOLDER_TO_SAVE_RESULTS>```

#### File descriptions

model.py - contains keras implementation of the custom Unet model described in the Noise2Noise paper

train.py - trains the Unet model

test.py - produces results on new images by running the Unet model with the learned weights

generator.py - generates bacthes of training and validation datasets to feed into the network

preprocess_images.py - contains methods for reading, preprocessing and saving data as numpy arrays:
* read data from .nd2 files
* align, augment, and clean the data
* save data as numpy array
