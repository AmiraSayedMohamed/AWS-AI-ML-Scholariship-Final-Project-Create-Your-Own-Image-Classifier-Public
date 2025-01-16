import torch
from PIL import Image
import json
import argparse
from src.utils import process_image
from src.model import load_model

# Command-line arguments
parser = argparse.ArgumentParser(description='Predict the class of an image')
parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
args = parser.parse_args()

# Load class-to-name mapping
with open('/content/flower_data/class_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Load model
model_path = '/content/checkpoints/model_epoch_5.pth'  # Update with actual checkpoint path
model = load_model(model_path, num_classes=102)

# Image preprocessing
image = process_image(args.image_path).to('cuda' if torch.cuda.is_available() else 'cpu')

# Model Prediction
model.eval()
with torch.no_grad():
    output = model(image)
    _, predicted_class = torch.max(output, 1)

# Get the class name
class_name = cat_to_name[str(predicted_class.item())]
print(f"Predicted class: {class_name}")
