import os
import torch
import gdown
from timm import create_model

def load_model():
    model_path = "models/swin_transformer_trained2225.pth"
    os.makedirs("models", exist_ok=True)

    # Google Drive URL from your shared file
    if not os.path.exists(model_path):
        print("Downloading Swin Transformer model from Google Drive...")
        url = "https://drive.google.com/uc?id=1YKsKLsK0P8_m05GRBEZUrMFsjKli5zKx"
        gdown.download(url, model_path, quiet=False)

    # Load Swin Transformer
    model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model
