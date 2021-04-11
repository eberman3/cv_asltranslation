"""
Project 6 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

from tensorflow.keras import models

import hyperparameters as hp


class ASLModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(ASLModel, self).__init__()

        self.optimizer = None #fix

        self.architecture = []

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)

        loss_fn = None #fix
        return loss_fn(labels, predictions)


