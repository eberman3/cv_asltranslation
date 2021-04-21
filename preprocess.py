"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

import hyperparameters as hp

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, architecture):

        self.data_path = data_path
        self.architecture = architecture

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Mean and std for standardization
        self.mean = np.zeros((3,))
        self.std = np.zeros((3,))
        self.calc_mean_and_std()

        # Setup data generators
        ds_asl_dir = data_path
        data_gen, self.train_data, self.test_data = self.get_data(path=ds_asl_dir,
         shuffle=True, augment=True)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path)):
            for name in files:
                if name.endswith(".jpeg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        img_size = 64
        if self.architecture == 'ASL':
            img_size = 64
        elif self.architecture == 'VGG':
            img_size = 224
        elif self.architecture == 'AlexNet':
            img_size = 256
        elif self.architecture == 'LeNet':
            img_size = 28

        data_sample = np.zeros(
            (hp.preprocess_sample_size, img_size, img_size, 3))
        print(data_sample.size)


        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((img_size, img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

        self.mean[0] = np.mean(data_sample[:,:,:,0])
        self.mean[1] = np.mean(data_sample[:,:,:,1])
        self.mean[2] = np.mean(data_sample[:,:,:,2])
        self.std[0] = np.std(data_sample[:,:,:,0])
        self.std[1] = np.std(data_sample[:,:,:,1])
        self.std[2] = np.std(data_sample[:,:,:,2])

        # ==========================================================

        print("Dataset mean: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0], self.mean[1], self.mean[2]))

        print("Dataset std: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0], self.std[1], self.std[2]))
        print(self.architecture)

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """

        img = (img - self.mean)/self.std
        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """

        if self.architecture == 'VGG':
            print("Now doing VGG Preprocessing.")
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            img = img / 255.
            img = self.standardize(img)
        return img

    def get_data(self, path, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        if augment:

            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
                rotation_range=15, 
                rescale=1/255, 
                zoom_range=0.1, 
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                validation_split=0.2)
        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # setting up different sizes
        img_size = 64
        if self.architecture == 'ASL':
            img_size = 64
        elif self.architecture == 'VGG':
            img_size = 224
        elif self.architecture == 'AlexNet':
            img_size = 256
        elif self.architecture == 'LeNet':
            img_size = 28


        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        train_data = data_gen.flow_from_directory(directory=path, target_size=(img_size, img_size),
                                                     class_mode="sparse", batch_size=hp.batch_size, classes=classes_for_flow, subset="training")
        test_data = data_gen.flow_from_directory(directory=path, target_size=(img_size, img_size),
                                                    class_mode="sparse", batch_size=hp.batch_size, classes=classes_for_flow, subset="validation")

        data_gen = train_data
        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)
            print(unordered_classes)
            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen, train_data, test_data
