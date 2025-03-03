import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import pickle
import io

app = Flask(__name__)

class FastNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = self.a1 @ self.W2 + self.b2
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Load Model Safely
MODEL_PATH = "model.pkl"
model, label_encoder = None, None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as file:
            model, label_encoder = pickle.load(file)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model not loaded!'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded!'}), 400

    try:
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB').resize((128, 128))
        img_array = np.array(image).flatten() / 255.0  # Normalize

        if img_array.shape[0] != 128 * 128 * 3:  # Ensure correct shape
            return jsonify({'error': 'Invalid image size!'}), 400

        img_array = np.expand_dims(img_array, axis=0)  # Convert to 2D array
        prediction = model.predict(img_array)
        predicted_species = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'predicted_species': predicted_species})
    except Exception as e:
        print(f"❌ Error in prediction: {e}")  # Print error in logs
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render’s PORT variable
    app.run(host="0.0.0.0", port=port, debug=True)  # Enable Debug Mode
