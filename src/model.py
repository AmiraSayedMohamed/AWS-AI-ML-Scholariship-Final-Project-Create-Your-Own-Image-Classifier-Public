import torch
from torchvision import models, transforms
import torch.nn as nn

def create_model(num_classes):
    """
    Createing a ResNet152 model and adjusting the last layer for the number of classes in the dataset.
    """
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(model_path, num_classes):
    """
    Loading a trained model from the checkpoint.
    """
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
