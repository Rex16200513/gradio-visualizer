import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os

# 自定义黑夜主题
class TrueDarkTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.blue,
            secondary_hue=colors.gray,
            neutral_hue=colors.gray,
            font=fonts.GoogleFont("Inter"),
            dark=True
        )

theme = TrueDarkTheme()

# 读取标签文件
with open('label.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(weights=None)  # 不加载预训练权重

# 定义模型结构（确保与训练时一致）
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(class_labels))
)

# 加载训练好的模型权重
model.load_state_dict(torch.load("best_model.pth", map_location=device), strict=False)
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
    return f"Predicted class: {class_labels[predicted_idx]} (Confidence: {probabilities[predicted_idx]:.2f})"

# 创建 Gradio 接口
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Text(label="Prediction Result"),
    theme=theme
)

# 启动服务
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 5000)))
