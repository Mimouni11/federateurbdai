import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_gradcam_image(model, pil_image: Image.Image, device: str) -> Image.Image:
    """
    Takes a PIL image, runs Grad-CAM using the last conv block of EfficientNet-B0,
    returns a PIL image with the heatmap overlay.
    """
    tensor        = TRANSFORM(pil_image).unsqueeze(0).to(device)
    img_resized   = pil_image.resize((224, 224))
    img_array     = np.array(img_resized).astype(np.float32) / 255.0
    target_layers = [model.features[-1]]

    cam           = GradCAM(model=model, target_layers=target_layers)

    # pos_label=0 because ImageFolder assigns fake=0 (alphabetical)
    # We want Grad-CAM to highlight regions that activate for the fake class
    targets       = [BinaryClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)