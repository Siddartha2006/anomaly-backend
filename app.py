from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import your swin model
from models.swin_model import load_model as load_swin_model

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
GRADCAM_FOLDER = 'gradcam_outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

# --- MongoDB Setup ---
client = MongoClient('mongodb+srv://new-user:Sairohan890@cluster0.2zowk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['anomaly_detection']
users = db['users']

# --- Load models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Swin Transformer for classification
classifier_model = load_swin_model()
classifier_model.to(device)
classifier_model.eval()

# ResNet18 for GradCAM visualization only
gradcam_model = resnet18(weights=ResNet18_Weights.DEFAULT)
gradcam_model.eval()
gradcam_model.to(device)
target_layers = [gradcam_model.layer4[-1]]

# --- Class Labels ---
class_labels = ['defective', 'non-defective']

# --- API Endpoints ---

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if users.find_one({'username': username}):
        return jsonify({'status': 'user_exists'})

    users.insert_one({'username': username, 'password': password})
    return jsonify({'status': 'registered'})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = users.find_one({'username': username, 'password': password})
    if user:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'fail'})


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Load and preprocess image
    img_pil = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Grad-CAM (use ResNet model only for CAM)
    cam = GradCAM(model=gradcam_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    rgb_img = np.array(img_pil.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    gradcam_filename = 'gradcam_' + filename
    gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
    cv2.imwrite(gradcam_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    # Swin Transformer classification
    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    predicted_label = class_labels[predicted_class] if predicted_class < len(class_labels) else 'unknown'

    return jsonify({
        'uploaded_image_url': f'http://127.0.0.1:5000/uploads/{filename}',
        'gradcam_image_url': f'http://127.0.0.1:5000/gradcam_outputs/{gradcam_filename}',
        'predicted_label': predicted_label,
        'confidence_score': f"{confidence:.2f}"
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/gradcam_outputs/<filename>')
def gradcam_file(filename):
    return send_from_directory(app.config['GRADCAM_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
