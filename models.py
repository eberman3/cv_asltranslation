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

        self.optimizer = self.optimizer = tf.keras.optimizers.Adam(lr=hp.learning_rate) #fix

        self.architecture = [
            # Block 1
            Conv2D(16,(3,3),padding="same",activation="relu",input_shape=((hp.img_size, hp.img_size, 3))),
            MaxPool2D(2,2),
            Dropout(0.25),
            Conv2D(32,(3,3),padding="same",activation="relu"),
            Conv2D(32,(3,3),padding="same",activation="relu"),
            MaxPool2D(2,2),
            Dropout(0.2),
            Conv2D(64,(3,3),padding="same",activation="relu"),
            Conv2D(64,(3,3),padding="same",activation="relu"),
            MaxPool2D(3,3),
            Conv2D(128,(3,3),padding="same",activation="relu"),
            MaxPool2D(2,2),
            Dropout(0.2),
            BatchNormalization(),
            Flatten(),
            Dense(128,activation="relu"),
            Dropout(0.2),
            Dense(64,activation="relu"),
            Dense(hp.num_classes,activation="softmax")
        ]

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

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss_fn(labels, predictions)


