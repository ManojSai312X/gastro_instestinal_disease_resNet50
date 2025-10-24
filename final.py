from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load the pre-trained model from specific path
def load_trained_model():
    try:
        model_path = 'resNet.h5'
        
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model = load_trained_model()

# Class labels
class_labels = [
    "dyed-lifted-polyps", 
    "dyed-resection-margins", 
    "esophagitis", 
    "normal-cecum", 
    "normal-pylorus", 
    "normal-z-line", 
    "polyps",
    "ulcerative-colitis"
]

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to make prediction
def predict_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)
    return predicted_class, confidence, predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server configuration.'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read and process the image
        image_data = Image.open(file.stream)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image_data.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Make prediction
        processed_image = preprocess_image(image_data)
        predicted_class, confidence, all_predictions = predict_image(model, processed_image)
        
        # Prepare results
        result = {
            'predicted_class': class_labels[predicted_class[0]],
            'confidence': float(confidence) * 100,
            'image_data': f"data:image/jpeg;base64,{img_str}",
            'all_predictions': {label: float(pred) * 100 for label, pred in zip(class_labels, all_predictions[0])}
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

# Health check endpoint to verify model loading
@app.route('/health')
def health_check():
    if model is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    app.run(debug=True)