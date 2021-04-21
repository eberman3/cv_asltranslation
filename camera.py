from cv2 import cv2
import numpy as np

from models import ASLModel
import hyperparameters as hp
import tensorflow as tf
import os

def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        frame = frame[start: start + w]
    else:
        frame = frame[:, start: start + h]
    return frame

def session(model):
    
    index_to_letter = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    mean = 0.485 * 225.
    std = 0.229 * 225.

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        frame = center_crop(frame)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        #img = preprocessing.image.load_img("asl_dataset/asl_dataset/a/hand1_a_bot_seg_1_cropped.jpeg", target_size=(64, 64))
        #x = preprocessing.image.img_to_array(img)
        x = cv2.resize(frame, (64, 64))
        x = (x - mean) / std

        x = np.array(x)
        x = np.expand_dims(x, axis=0)   

        image = np.vstack([x])
        classes = model.predict(image, batch_size=1)
        index = np.argmax(classes,axis=1)
 

        print(classes) #displaying matrix prediction position
        print(index)
        letter = index_to_letter[int(index)]

        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("ASL Translation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()