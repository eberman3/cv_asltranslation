from cv2 import cv2
import numpy as np
import onnx
import onnxruntime as ort
import keras2onnx

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
    output_model_path = "keras_efficientNet.onnx"
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras2onnx.save_model(onnx_model, output_model_path)
    
    index_to_letter = list('ABCDEFGHIJKLMNOPQRSTUVWXY')
    mean = 0.485 * 225.
    std = 0.229 * 225.

    ort_session = ort.InferenceSession(output_model_path)
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        frame = center_crop(frame)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(frame, (64, 64))
        x = (x - mean) / std

        x = x.reshape(1, 64, 64, 3).astype(np.float32)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        y = ort_session.run([output_name], {input_name: x})[0]

        index = np.argmax(y, axis=1)
        print(index)
        letter = index_to_letter[int(index)]

        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("ASL Translation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()