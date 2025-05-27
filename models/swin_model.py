import torch
import timm

def load_model():
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load('models/swin_transformer_trained_earlystop.pth', map_location=torch.device('cpu')))
    model.eval()
    return model
