import gradio as gr
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os

# ===== 自定义真正的黑夜主题 =====
import gradio.themes.base as base

class TrueDarkTheme(base.Base):
    def __init__(self):
        super().__init__(
            primary_hue="blue",
            neutral_hue="slate",
            font=[base.GoogleFont("Inter"), "system-ui", "sans-serif"],
            spacing_size="sm",
            radius_size="md",
            text_size="md",
            dark=True,
            colors={
                "background": "#111111",
                "text": "#FFFFFF",
                "button-primary-background": "#333333",
                "button-primary-text": "#FFFFFF",
                "input-background": "#222222",
                "input-text": "#FFFFFF",
            }
        )

theme = TrueDarkTheme()

# ===== 读取标签文件 =====
with open('label.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# ===== 加载模型 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(weights=None)

# 修改全连接层
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(class_labels))
)

# 加载权重
model.load_state_dict(torch.load("best_model.pth", map_location=device), strict=False)
model.eval()

# 图像预处理
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# ===== 预测函数 =====
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
    return f"Predicted class: {class_labels[predicted_idx]} (Confidence: {probabilities[predicted_idx]:.2f})"

# ===== Gradio 接口 =====
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Text(label="Prediction Result"),
    theme=theme,
    title="Nature Interaction Classifier",
    description="Upload an image to classify the type of human-nature interaction behavior it represents.",
    article="Trained on ResNet50 with human-nature interaction dataset collected from social media."
)

# ===== 启动服务 =====
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 5000)))
