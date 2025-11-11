import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Model & Config
MODEL_PATH = "best_parkinson_model.h5"  # or "parkinson_detector.h5"
IMG_SIZE = 128
CATEGORIES = ["Healthy", "Parkinson"]

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print(" Model loaded successfully!")

# Preprocess and predict
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image file.")

    # Convert BGR → RGB (MobileNetV2 expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # same normalization as training

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    result = CATEGORIES[class_index]
    return result, confidence

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            os.makedirs("static", exist_ok=True)
            image_path = os.path.join("static", file.filename)
            file.save(image_path)

            try:
                result, confidence = predict_image(image_path)
                confidence = round(confidence * 100, 2)
            except Exception as e:
                result = f"Error: {e}"
                confidence = None

    return render_template('index.html', result=result, confidence=confidence, image_path=image_path)

# Run Flask
if __name__ == '__main__':
    app.run(debug=True)
