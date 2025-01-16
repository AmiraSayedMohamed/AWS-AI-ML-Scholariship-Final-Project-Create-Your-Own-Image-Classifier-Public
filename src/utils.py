# This file contain helper function such as image processing
from torchvision import transforms
from PIL import Image
import torch

def get_transform():
    """
    Returning the transformation pipeline for images.
    """
    return transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def process_image(image_path):
    """
    Processing an image and returns the transformed tensor.
    """
    image = Image.open(image_path)
    transform = get_transform()
    return transform(image).unsqueeze(0) # Add batch dimension
