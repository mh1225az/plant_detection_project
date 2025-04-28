import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

model = tf.keras.models.load_model('models/plant_model.h5')
class_names = sorted(os.listdir('dataset'))

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img) / 255.
    return np.expand_dims(img_array, axis=0)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    pred = model.predict(preprocess(frame))
    label = class_names[np.argmax(pred)]

    cv2.putText(frame, f'Tanaman: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Deteksi Tanaman", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
