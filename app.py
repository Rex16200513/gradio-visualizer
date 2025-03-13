import gradio as gr
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# 读取标签文件
with open('label.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_labels))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 图像预处理
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 预测函数
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
    return f"预测类别: {class_labels[predicted_idx]} (置信度: {probabilities[predicted_idx]:.2f})"

# Gradio 接口
iface = gr.Interface(fn=predict_image, inputs=gr.Image(type="pil"), outputs="text")
iface.launch()
