from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # sesuaikan ukuran dengan model
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalisasi
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
