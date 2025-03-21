import gradio as gr
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os

# 读取标签文件
with open('label.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(weights=None)  # 不加载预训练权重

# 定义模型的全连接层（确保与训练时一致）
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),  # 第一层
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(class_labels))  # 根据标签数量调整输出层
)

# 加载训练好的模型权重
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()  # 设置为评估模式

# 图像预处理
transform = Compose([
    Resize((256, 256)),  # 调整图片大小
    ToTensor(),  # 转换为Tensor
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 标准化
])

# 预测函数
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)  # 对图像进行预处理，并增加批次维度
    with torch.no_grad():  # 禁用梯度计算，减少内存消耗
        outputs = model(image)  # 进行预测
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]  # 计算预测的类别概率
        predicted_idx = torch.argmax(probabilities).item()  # 获取最大概率的索引
    return f"预测类别: {class_labels[predicted_idx]} (置信度: {probabilities[predicted_idx]:.2f})"

# Gradio 接口
iface = gr.Interface(fn=predict_image, inputs=gr.Image(type="pil"), outputs="text")

# 启动 Gradio 服务并绑定端口
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 5000)))
