import os
import torch
import gdown
from timm import create_model

def load_model():
    model_dir = "models"
    model_path = os.path.join(model_dir, "swin_transformer_trained2225.pth")

    os.makedirs(model_dir, exist_ok=True)

    gdrive_url = "https://drive.google.com/uc?id=14ATiSazoF1SH2bf900A9iQbP2DTSiuky"

    if not os.path.exists(model_path):
        print("Downloading Swin Transformer model from Google Drive...")
        gdown.download(gdrive_url, model_path, quiet=False)

    model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully.")
    return model

