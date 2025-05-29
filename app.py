from datetime import datetime
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
from models.swin_model import load_model as load_swin_model

app = Flask(__name__)

# --- CORS configuration for new Vercel frontend ---
CORS(app, resources={r"/api/*": {
    "origins": [
        "https://anomaly-frontend-r9wualvma-siddartha2006s-projects.vercel.app",
        "http://localhost:3000"
    ]
}}, supports_credentials=True)

# Upload folders
UPLOAD_FOLDER = 'uploads'
GRADCAM_FOLDER = 'gradcam_outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

# MongoDB setup
client = MongoClient(
    'mongodb+srv://new-user:Sairohan890@cluster0.2zowk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
)
db = client['anomaly_detection_new']
users = db['users']
history_collection = db['history']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
classifier_model = load_swin_model()
classifier_model.to(device)
classifier_model.eval()

gradcam_model = resnet18(weights=ResNet18_Weights.DEFAULT)
gradcam_model.eval()
gradcam_model.to(device)
target_layers = [gradcam_model.layer4[-1]]

class_labels = ['defective', 'non-defective']

# --- API endpoints ---

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    if users.find_one({'username': username}):
        return jsonify({'status': 'user_exists'}), 400

    users.insert_one({'username': username, 'password': password})
    return jsonify({'status': 'registered'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    user = users.find_one({'username': username, 'password': password})
    if user:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'fail'}), 401

@app.route('/api/upload', methods=['POST'])
def upload_image():
    username = request.form.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Image preprocessing
    img_pil = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Grad-CAM generation
    cam = GradCAM(model=gradcam_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    rgb_img = np.array(img_pil.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    gradcam_filename = 'gradcam_' + filename
    gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
    cv2.imwrite(gradcam_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    # Classification
    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    predicted_label = class_labels[predicted_class] if predicted_class < len(class_labels) else 'unknown'

    # Save history record to MongoDB
    record = {
        'username': username,
        'image_url': f'http://127.0.0.1:5000/uploads/{filename}',
        'gradcam_url': f'http://127.0.0.1:5000/gradcam_outputs/{gradcam_filename}',
        'label': predicted_label,
        'confidence': confidence,
        'timestamp': datetime.utcnow()
    }
    history_collection.insert_one(record)

    return jsonify({
        'uploaded_image_url': record['image_url'],
        'gradcam_image_url': record['gradcam_url'],
        'predicted_label': predicted_label,
        'confidence_score': f"{confidence:.2f}"
    })

@app.route('/api/user-history/<username>', methods=['GET'])
def get_user_history(username):
    try:
        records_cursor = history_collection.find({'username': username}).sort('timestamp', -1)
        records = list(records_cursor)
        for record in records:
            record['_id'] = str(record['_id'])
            record['timestamp'] = record['timestamp'].isoformat()
        return jsonify({
            'status': 'success',
            'username': username,
            'history': records
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    username = request.args.get('username', None)
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)

    query = {}
    if username:
        query['username'] = username

    if start_date or end_date:
        date_query = {}
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                date_query['$gte'] = start_dt
            except ValueError:
                return jsonify({'error': 'Invalid start_date format. Use YYYY-MM-DD'}), 400
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                date_query['$lte'] = end_dt
            except ValueError:
                return jsonify({'error': 'Invalid end_date format. Use YYYY-MM-DD'}), 400

        if date_query:
            query['timestamp'] = date_query

    records_cursor = history_collection.find(query).sort('timestamp', -1)
    records = list(records_cursor)

    for record in records:
        record['_id'] = str(record['_id'])
        record['timestamp'] = record['timestamp'].isoformat()

    grouped = {}
    for record in records:
        user = record['username']
        if user not in grouped:
            grouped[user] = []
        grouped[user].append({
            'image_url': record['image_url'],
            'gradcam_url': record['gradcam_url'],
            'label': record['label'],
            'confidence': record['confidence'],
            'timestamp': record['timestamp']
        })

    return jsonify(grouped)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/gradcam_outputs/<filename>')
def gradcam_file(filename):
    return send_from_directory(app.config['GRADCAM_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

