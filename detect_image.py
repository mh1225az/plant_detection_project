import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('models/plant_model.h5')
class_names = sorted(os.listdir('dataset'))

# Fungsi preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Loop semua gambar di dataset  
def detect_from_dataset():
    for class_dir in class_names:
        folder_path = os.path.join('dataset', class_dir)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                img_array = preprocess_image(img_path)
                prediction = model.predict(img_array)[0]
                predicted_idx = np.argmax(prediction)
                predicted_label = class_names[predicted_idx]
                confidence = prediction[predicted_idx]

                print(f"{file} | Label Asli: {class_dir} | Prediksi: {predicted_label} ({confidence*100:.2f}%)")

                # Tampilkan gambar dengan label prediksi
                img_cv = cv2.imread(img_path)
                cv2.putText(img_cv, f'{predicted_label} ({confidence*100:.1f}%)', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Hasil Deteksi", img_cv)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error memproses {img_path}: {e}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_dataset()
