# 🌼 Custom Flower Image Classifier 🌼
![Flower Classification Result](https://github.com/AmiraSayedMohamed/AWS-AI-ML-Scholariship-Final-Project-Create-Your-Own-Image-Classifier-Public/blob/master/flower_data/output-Img.jpg)
Welcome to **Custom Flower Image Classifier**, a project developed as part of the **AWS AI & ML Scholarship** program. This project implements a deep learning-based image classification model capable of identifying different types of flowers from images. 🚀

---

## 🎯 Project Overview

The goal of this project is to build a custom image classifier using a **pretrained deep learning model** (e.g., VGG16 or VGG13) from PyTorch's `torchvision.models`. The project trains the model on a flower dataset, validates its performance, and uses it to predict the flower species of a given image.

The project is divided into two main components:
1. **Training**: Train the model on a labeled dataset of flower images.
2. **Prediction**: Use the trained model to predict flower species from new images.

## Project Structure

- `flower_data/`: Contains the dataset including training, validation, and test images.
- `checkpoints/`: Directory to save model checkpoints during training.
- `src/`: Contains all code files including model architecture, training, prediction, and helper functions.
- `requirements.txt`: Lists dependencies needed to run the project.

## Setup

1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Ensure that the `flower_data/` folder contains the `train/`, `valid/`, and `test/` directories.

3. To train the model, run:
    ```bash
    python src/train.py
    ```

4. To predict with the trained model, run:
    ```bash
    python src/predict.py --image_path <path_to_image>
    ```

## Notes

- The flower names are manually generated since the original JSON file was missing. Therefore, the names might not correspond to the actual flower names.
