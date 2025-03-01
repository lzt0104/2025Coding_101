import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
import os
import pyttsx3
import time
from PIL import Image, ImageDraw, ImageFont
import random

# 初始化 pyttsx3 引擎（非同步模式）
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 語速
engine.setProperty('volume', 1.0)  # 音量
engine.startLoop(False)  # 啟動非同步語音播放模式

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        return x.view(x.size(0), -1)

class CNNLSTMModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size):
        super(CNNLSTMModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.lstm = nn.LSTM(1280, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, lengths):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        packed_input = nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        ht = torch.cat((ht[-2], ht[-1]), dim=1)
        out = self.fc(ht)
        return out

hidden_size = 128
num_layers = 1
output_size = 1

model = CNNLSTMModel(hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('modle/modle.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    return image

def predict_single_frame(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)
        lengths = torch.tensor([1], dtype=torch.int64).cpu()
        outputs = model(image_tensor, lengths)
        predicted = torch.sigmoid(outputs).item()
    return predicted

def draw_text_chinese(frame, text, position, font_path="C:/Windows/Fonts/msjh.ttc", font_size=60):
    """
    在 OpenCV 圖片上繪製中文字
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"字體檔案無法找到: {font_path}")
        return frame
    color = (255, 0, 0) if text == "請減速慢行" else (0, 255, 0)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# **非同步語音播報**
def speak_message(message):
    """
    非同步語音播放，確保即時性
    """
    engine.say(message)
    engine.iterate()  # 讓 pyttsx3 繼續處理隊列

# 處理影像的主迴圈
for i in range(12):
    # randint_ = [27, 118, 140, 14, 3, 32, 150, 155, 81, 54, 160, 89, 126]
    randint_ = random.sample(range(0, 179), 12)
    image_folder = f"train/train/freeway_{randint_[i]:04d}"

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('.')[0]))

    tf = 0
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        image_tensor = preprocess_image(img_path)

        if not tf:
            prediction = predict_single_frame(model, image_tensor)
            if prediction > 0.5:
                tf = 1
                speak_message("請減速慢行")  # 語音與文字同步
                frame = draw_text_chinese(frame, "請減速慢行", (10, 30))

        accident_prediction = "請減速慢行" if tf else "安全"
        frame = draw_text_chinese(frame, accident_prediction, (10, 30))
        cv2.imshow("image", frame)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

# 停止 pyttsx3 非同步模式
engine.endLoop()