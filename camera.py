"""
Project 6 - Loss in Translation
CS1430 - Computer Vision
Brown University
"""

from cv2 import cv2
import numpy as np

from models import ASLModel
import hyperparameters as hp
import tensorflow as tf
import os
from skimage.io import imread
from skimage.transform import resize

def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        frame = frame[start: start + w]
    else:
        frame = frame[:, start: start + h]
    return frame

def session(model, datasets):
    
    index_to_letter = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        frame = center_crop(frame)

        img = imread("asl_dataset/asl_dataset/b/hand1_b_bot_seg_1_cropped.jpeg")

        x = datasets.preprocess_fn(frame)
        x = resize(x, (64, 64, 3))
        x = x / 255
        x = np.expand_dims(x, axis=0)

        classes = model.predict(x, batch_size=1)
        index = np.argmax(classes,axis=1)
 
        letter = index_to_letter[int(index)]

        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("ASL Translation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()