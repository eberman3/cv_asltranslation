"""
Project 6 - CNNs
CS1430 - Computer Vision
Brown University
"""

"""
Number of epochs. 
"""
num_epochs = 30


learning_rate = 1e-4


img_size = 64

"""
Sample size for calculating the mean and standard deviation of the
training data. This many images will be randomly seleted to be read
into memory temporarily.
"""
preprocess_sample_size = 450

"""
Maximum number of weight files to save to checkpoint directory. If
set to a number <= 0, then all weight files of every epoch will be
saved. Otherwise, only the weights with highest accuracy will be saved.
"""
max_num_weights = 5

"""
Defines the number of training examples per batch.
"""
batch_size = 32

"""
The number of classes.
"""
num_classes = 36
