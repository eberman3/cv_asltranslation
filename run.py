"""
Project 6 - Loss in Translation
CS1430 - Computer Vision
Brown University
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras import preprocessing
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, AveragePooling2D
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
import tensorflow.keras
import hyperparameters as hp
from tensorflow.keras import regularizers
from camera import session
import hyperparameters as hp
from models import ASLModel, VGGModel, AlexNetModel, LeNetModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--architecture',
        default = 'ASL',
        choices=['ASL', 'VGG', 'AlexNet', 'LeNet'],
        help='''Which architecture to run'''
    )
    parser.add_argument(
        '--data',
        default = 'asl_dataset' + os.sep + 'asl_dataset' + os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--video',
        action='store_true',
        help='''Skips training and runs the live video predictor.
        You can use this to predict with an already trained model by loading
        its checkpoint.''')


    return parser.parse_args()

def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.architecture, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    print(init_epoch)

    earlystop=tf.keras.callbacks.EarlyStopping(patience=10)
    learning_rate_reduce=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_sparse_categorical_accuracy",min_lr=0.001)
    callback_list.append(earlystop)
    callback_list.append(learning_rate_reduce)
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = Datasets(ARGS.data, ARGS.architecture)

    if ARGS.architecture == 'ASL':
        model = ASLModel()
        img_size = 64
        model(tf.keras.Input(shape=(img_size, img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "ASLModel" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "ASLModel" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    elif ARGS.architecture == 'VGG':
        model = VGGModel()
        img_size = 224
        model(tf.keras.Input(shape=(img_size, img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "VGGModel" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "VGGModel" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    elif ARGS.architecture == 'AlexNet':
        model = AlexNetModel()
        img_size = 256
        model(tf.keras.Input(shape=(img_size, img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "AlexNetModel" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "AlexNetModel" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    elif ARGS.architecture == 'LeNet':
        model = LeNetModel()
        img_size = 28
        model(tf.keras.Input(shape=(img_size, img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "LeNetModel" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "LeNet" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])


    if ARGS.evaluate:
        test(model, datasets.test_data)
    elif ARGS.video:
        session(model, datasets)
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)


# Make arguments global
ARGS = parse_args()

main()

