import json
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import os
import base64
import tensorflow as tf

app = Flask(__name__)

model_path = 'model.h5'
model = load_model(model_path)

class_indices_path = 'class_indices.json'
with open(class_indices_path, 'r') as json_file:
    class_indices = json.load(json_file)
    class_names = {v: k for k, v in class_indices.items()}

last_uploaded_image = None

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global last_uploaded_image  # Global, used in multiple methods

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    file_path, img_array = get_img_array_from_file(file)

    # Predict using the loaded model
    predictions = model.predict(img_array)
    top_1_predicted_class_index, top_1_predicted_class_name, top_3_predictions = get_top_predictions(predictions)

    # convert the image to base64 for display
    last_uploaded_image = base64.b64encode(open(file_path, 'rb').read()).decode('utf-8')

    return jsonify({
        'predicted_class': top_1_predicted_class_name,
        'probability': float(predictions[0][top_1_predicted_class_index]),
        'top3_predictions': top_3_predictions,
        'uploaded_image': last_uploaded_image
    })

def get_top_predictions(predictions):
    top_1_predicted_class_index = np.argmax(predictions[0])
    top_1_predicted_class_name = class_names.get(top_1_predicted_class_index, f'Class_{top_1_predicted_class_index}')
    top_3_predicted_class_indexes = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {'class': class_names.get(idx, f'Class_{idx}'), 'probability': float(predictions[0][idx])}
        for idx in top_3_predicted_class_indexes
    ]
    return top_1_predicted_class_index, top_1_predicted_class_name, top_3_predictions

def get_img_array_from_file(file):
    # save the uploaded file to a temporary location
    upload_folder = '/tmp/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # resize image for prediction
    img = Image.open(file_path)
    img = img.resize((224, 224))

    # Need to apply same preprocessing used during training
    augment = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.15),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.15)
    ])

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = augment(np.expand_dims(img_array, axis=0))[0]
    img_array /= 1  # Normalize
    # Ensure the image array has the correct shape (add an extra dimension)
    img_array = np.expand_dims(img_array, axis=0)

    return file_path, img_array


@app.route('/display_image')
def display_image():
    global last_uploaded_image
    if last_uploaded_image is not None:
        return jsonify({'uploaded_image': last_uploaded_image})
    else:
        return jsonify({'error': 'No uploaded image'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
