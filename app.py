import gradio as gr
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os

# Read label file
with open('label.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(weights=None)  # Use latest API, replace 'pretrained=False'
model.fc = nn.Linear(model.fc.in_features, len(class_labels))  # Adjust the fully connected layer to match the number of labels
model.load_state_dict(torch.load("best_model.pth", map_location=device))  # Load the trained model weights
model.eval()  # Set model to evaluation mode

# Image preprocessing
transform = Compose([
    Resize((256, 256)),  # Resize the image
    ToTensor(),  # Convert to Tensor
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize
])

# Prediction function
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)  # Preprocess the image and add batch dimension
    with torch.no_grad():  # Disable gradient calculation to save memory
        outputs = model(image)  # Make prediction
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]  # Calculate class probabilities
        predicted_idx = torch.argmax(probabilities).item()  # Get the index of the highest probability
    return f"Predicted Class: {class_labels[predicted_idx]} (Confidence: {probabilities[predicted_idx]:.2f})"

# Gradio interface with custom text for image upload
iface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(type="pil", label="Upload an image to classify"),  # Custom label for image input
    outputs="text", 
    title="Image Classifier",  # Optional: you can add a title
    description="Upload an image, and the model will predict the class and confidence."
)

# Launch Gradio app and bind to port
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 5000)))
