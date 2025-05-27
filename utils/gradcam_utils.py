import torch
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(image_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        confidence, predicted_class = torch.max(torch.nn.functional.softmax(output, dim=1), 1)

    # GradCAM target layer (for SwinTransformer from timm)
    try:
        target_layer = model.layers[-1].blocks[-1].norm1
    except AttributeError:
        raise Exception("Could not find the correct target layer. Please check model architecture.")

    # Generate Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(predicted_class.item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Overlay CAM on original image
    rgb_img = np.array(img.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save image
    result_path = image_path.replace("uploads", "static")
    os.makedirs("static", exist_ok=True)
    cv2.imwrite(result_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return {
        "status": "success",
        "prediction": str(predicted_class.item()),
        "confidence": float(confidence.item()),
        "cam_path": result_path.replace("\\", "/")
    }
