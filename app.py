from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from utils.preprocessing import preprocess_image

app = Flask(__name__, static_folder='static')

# Load model
model = load_model('model/model_cabai.h5')

# Mapping label indeks ke nama penyakit
label_map = {
    0: 'Antraknosa (Patek)',
    1: 'Gemini Virus',
    2: 'Layu Fusarium'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    # Simpan file sementara
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Preprocessing dan prediksi
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = label_map.get(predicted_class, 'Tidak Dikenal')

    # Hapus file setelah prediksi
    os.remove(file_path)

    return jsonify({'penyakit': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
