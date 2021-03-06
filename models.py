"""
Project 6 - Loss in Translation
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, AveragePooling2D

from tensorflow.keras import models

import hyperparameters as hp

#ASL MODEL
class ASLModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(ASLModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=hp.learning_rate) 


        self.architecture = [
            Conv2D(64,(3,3),padding="same",activation="relu",input_shape=((64, 64, 3))),
            Conv2D(64,(3,3),padding="same",activation="relu",input_shape=((64, 64, 3))),
            MaxPool2D(3,3),
            Conv2D(128,(5,5),padding="same",activation="relu"),
            Conv2D(128,(5,5),padding="same",activation="relu"),
            MaxPool2D(3,3),
            Conv2D(256,(5,5),padding="same",activation="relu"),
            Conv2D(256,(5,5),padding="same",activation="relu"),
            MaxPool2D(3,3),
            Conv2D(512,(3,3),padding="same",activation="relu"),
            Conv2D(512,(3,3),padding="same",activation="relu"),
            MaxPool2D(2,2),
            BatchNormalization(),
            Dropout(0.2),
            Flatten(),
            Dropout(0.4),
            BatchNormalization(),
            Dense(1024,activation="relu"),
            Dense(512,activation="relu"),
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

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        # Don't change the below:

        self.architecture = [
            Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"),
            Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
            MaxPool2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2,2),strides=(2,2)),
            Flatten(),
            Dense(units=4096,activation="relu"),
            Dense(units=4096,activation="relu"),
            Dense(hp.num_classes, activation="softmax")
        ]


    def call(self, x):
        """ Passes the image through the network. """
        for layer in self.architecture:
           x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """
        # Sparse_categorical_crossentropy
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

class AlexNetModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(AlexNetModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=hp.learning_rate) 


        self.architecture =  [
            Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(256,256,3)),
            BatchNormalization(),
            MaxPool2D(pool_size=(3,3), strides=(2,2)),
            Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(3,3), strides=(2,2)),
            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            BatchNormalization(),
            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            BatchNormalization(),
            Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(3,3), strides=(2,2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(hp.num_classes, activation='softmax')
        ]
        

    def call(self, x):
        """ Passes input image through the network. """
        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss_fn(labels, predictions)

class LeNetModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(LeNetModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=hp.learning_rate) 


        self.architecture =  [
            Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,3)),
            AveragePooling2D(),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
            AveragePooling2D(),
            Flatten(),
            Dense(units=120, activation='relu'),
            Dense(units=84, activation='relu'),
            Dense(units=hp.num_classes, activation = 'softmax'),
        ]
        

    def call(self, x):
        """ Passes input image through the network. """
        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss_fn(labels, predictions)
