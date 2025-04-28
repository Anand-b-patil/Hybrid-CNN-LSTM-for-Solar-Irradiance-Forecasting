from flask import Flask, render_template, request, jsonify
from cnn_model import SolarCNN
import torch
import cv2
import numpy as np
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Initialize model once
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolarCNN().to(device)
    model.load_state_dict(torch.load('models/cnn_model.pth', map_location=device))
    model.eval()
    return model, device

model, device = load_model()

def process_image(image_bytes, device):
    """Process uploaded image to match model requirements"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize to 240x320 (H x W)
        img = cv2.resize(img, (320, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0  # normalize to [0, 1]
        img_tensor = torch.tensor(img).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        return img_tensor.to(device)
    
    except Exception as e:
        raise BadRequest(f"Image processing failed: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid file type'}), 400
        
        img_tensor = process_image(file.read(), device)

        with torch.no_grad():
            prediction = model(img_tensor).item()
        
        return jsonify({
            'irradiance': round(prediction, 2),
            'units': 'W/m²'
        })
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
