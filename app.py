# app.py - Flask web application for deepfake detection
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import numpy as np

# Import your model architecture
# Make sure this matches exactly how you defined it during training
from model import DeepfakeDetector  # Create this file with your model class definition

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = r'C:\Users\anoop\Desktop\deepfakeImageDetection\static\uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepfakeDetector().to(device)
    model.load_state_dict(torch.load(r'C:\Users\anoop\Desktop\deepfakeImageDetection\deepfake_detector_model.pth', map_location=device))
    model.eval()  # Set to evaluation mode
    return model, device

# Initialize model
model, device = load_model()

# Image preprocessing
def preprocess_image(image):
    # Use the same transformations as during testing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Open and preprocess the image
            image = Image.open(file_path).convert('RGB')
            input_tensor = preprocess_image(image)
            input_tensor = input_tensor.to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                prediction = torch.argmax(probabilities).item()
                
                # Get probabilities
                real_prob = probabilities[0].item() * 100
                fake_prob = probabilities[1].item() * 100
            
            result = {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': fake_prob if prediction == 1 else real_prob,
                'real_probability': real_prob,
                'fake_probability': fake_prob,
                'image_path': file_path
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)




