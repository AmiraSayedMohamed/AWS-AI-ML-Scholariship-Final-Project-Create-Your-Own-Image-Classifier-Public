import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from src.utils import get_transform
from src.model import create_model
from sklearn.model_selection import train_test_split
import shutil

# Paths
train_dir = '/content/flower_data/train'
valid_dir = '/content/flower_data/test'
checkpoint_dir = '/content/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Parameters
batch_size = 32
learning_rate = 0.003
epochs = 5

# Dataset and DataLoader
transform = get_transform()
train_data = datasets.ImageFolder(train_dir, transform=transform)
valid_data = datasets.ImageFolder(valid_dir, transform=transform)

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

# Model
model = create_model(len(train_data.classes))  # Adjust for the number of classes
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader)}")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
