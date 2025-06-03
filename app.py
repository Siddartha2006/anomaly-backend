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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import re

app = Flask(__name__)

# --- CORS configuration for new Vercel frontend ---
CORS(app, resources={r"/api/*": {
    "origins": [
        "https://anomaly-frontend-4t8f4ru9g-siddartha2006s-projects.vercel.app",
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

# Email configuration - UPDATE THESE WITH YOUR ACTUAL EMAIL SETTINGS
EMAIL_CONFIG = {
    'SMTP_SERVER': 'smtp.gmail.com',  # For Gmail
    'SMTP_PORT': 587,
    'EMAIL_ADDRESS': 'thotasiddartha649@gmail.com',  # Your email address
    'EMAIL_PASSWORD': 'decx zloo yala njed',    # Your app password (not regular password)
    'FROM_NAME': 'Anomaly Detection System'
}

# MongoDB setup
client = MongoClient(
    'mongodb+srv://new-user:Sairohan890@cluster0.2zowk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
)
db = client['anomaly_detection_new']
users = db['users']
history_collection = db['history']
contacts_collection = db['contacts']

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

# Email functions
def create_confirmation_email(username, email, message):
    """Create HTML email content for contact form confirmation"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; background-color: #f9f9f9; }}
            .message-box {{ background-color: white; padding: 15px; border-left: 4px solid #4CAF50; margin: 10px 0; }}
            .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Contact Form Confirmation</h1>
                <p>Thank you for reaching out to us!</p>
            </div>
            <div class="content">
                <h2>Hello {username},</h2>
                <p>We have successfully received your message and want to confirm that it has been logged in our system.</p>
                
                <div class="message-box">
                    <h3>Your Message:</h3>
                    <p>{message}</p>
                </div>
                
                <p><strong>What happens next?</strong></p>
                <ul>
                    <li>Our team will review your message within 24-48 hours</li>
                    <li>If a response is required, we'll get back to you at: <strong>{email}</strong></li>
                    <li>For urgent matters, please call our support line</li>
                </ul>
                
                <p>Thank you for using our Anomaly Detection System!</p>
            </div>
            <div class="footer">
                <p>This is an automated message. Please do not reply to this email.</p>
                <p>&copy; 2025 Anomaly Detection System. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def send_email_async(to_email, subject, html_content):
    """Send email asynchronously to avoid blocking the main thread"""
    def send_email():
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{EMAIL_CONFIG['FROM_NAME']} <{EMAIL_CONFIG['EMAIL_ADDRESS']}>"
            msg['To'] = to_email

            # Create HTML part
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Send email
            server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'])
            server.starttls()
            server.login(EMAIL_CONFIG['EMAIL_ADDRESS'], EMAIL_CONFIG['EMAIL_PASSWORD'])
            server.send_message(msg)
            server.quit()
            
            print(f"Confirmation email sent successfully to {to_email}")
            
        except Exception as e:
            print(f"Failed to send email to {to_email}: {str(e)}")

    # Run email sending in a separate thread
    email_thread = threading.Thread(target=send_email)
    email_thread.daemon = True
    email_thread.start()

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

# Enhanced Contact endpoint with email confirmation
@app.route('/api/contact', methods=['POST'])
def contact():
    try:
        data = request.json
        
        # Extract data from request
        username = data.get('username')
        email = data.get('email')
        phone = data.get('phone', '')
        message = data.get('message')

        # Validate required fields
        if not username or not email or not message:
            return jsonify({
                'status': 'fail', 
                'message': 'Name, email, and message are required fields'
            }), 400

        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return jsonify({
                'status': 'fail',
                'message': 'Please enter a valid email address'
            }), 400

        # Create contact record
        contact_record = {
            'username': username.strip(),
            'email': email.strip().lower(),
            'phone': phone.strip() if phone else '',
            'message': message.strip(),
            'timestamp': datetime.utcnow(),
            'status': 'new'
        }

        # Insert into database
        result = contacts_collection.insert_one(contact_record)
        
        if result.inserted_id:
            # Send confirmation email asynchronously
            subject = "Contact Form Confirmation - Anomaly Detection System"
            html_content = create_confirmation_email(username, email, message)
            send_email_async(email, subject, html_content)
            
            return jsonify({
                'status': 'success',
                'message': 'Your message has been sent successfully. A confirmation email has been sent to your email address.',
                'contact_id': str(result.inserted_id)
            }), 200
        else:
            return jsonify({
                'status': 'fail',
                'message': 'Failed to save your message. Please try again.'
            }), 500

    except Exception as e:
        print(f"Contact form error: {str(e)}")
        return jsonify({
            'status': 'fail',
            'message': 'An unexpected error occurred. Please try again later.'
        }), 500

# Test email endpoint (optional - for testing email functionality)
@app.route('/api/test-email', methods=['POST'])
def test_email():
    """Test endpoint to verify email configuration"""
    try:
        data = request.json
        test_email = data.get('email')
        
        if not test_email:
            return jsonify({'status': 'fail', 'message': 'Email is required'}), 400
            
        subject = "Test Email - Anomaly Detection System"
        html_content = """
        <html>
        <body>
            <h2>Email Configuration Test</h2>
            <p>If you receive this email, your email configuration is working correctly!</p>
            <p>Timestamp: """ + datetime.now().isoformat() + """</p>
        </body>
        </html>
        """
        
        send_email_async(test_email, subject, html_content)
        
        return jsonify({
            'status': 'success',
            'message': 'Test email sent successfully!'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'fail',
            'message': f'Test email failed: {str(e)}'
        }), 500

# Admin endpoint to get all contact messages
@app.route('/api/admin/contacts', methods=['GET'])
def get_contacts():
    try:
        status = request.args.get('status', None)
        limit = int(request.args.get('limit', 50))
        
        query = {}
        if status:
            query['status'] = status
            
        contacts_cursor = contacts_collection.find(query).sort('timestamp', -1).limit(limit)
        contacts = list(contacts_cursor)
        
        for contact in contacts:
            contact['_id'] = str(contact['_id'])
            contact['timestamp'] = contact['timestamp'].isoformat()
            
        return jsonify({
            'status': 'success',
            'contacts': contacts,
            'count': len(contacts)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'fail',
            'message': str(e)
        }), 500

# Endpoint to update contact status
@app.route('/api/admin/contacts/<contact_id>/status', methods=['PUT'])
def update_contact_status(contact_id):
    try:
        data = request.json
        new_status = data.get('status')
        
        if not new_status:
            return jsonify({'status': 'fail', 'message': 'Status is required'}), 400
            
        from bson import ObjectId
        result = contacts_collection.update_one(
            {'_id': ObjectId(contact_id)},
            {'$set': {'status': new_status, 'updated_at': datetime.utcnow()}}
        )
        
        if result.modified_count > 0:
            return jsonify({'status': 'success', 'message': 'Status updated successfully'}), 200
        else:
            return jsonify({'status': 'fail', 'message': 'Contact not found'}), 404
            
    except Exception as e:
        return jsonify({'status': 'fail', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

