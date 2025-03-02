import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
import os
from random import randint
import pyttsx3  # 確保安裝了 pyttsx3
# import RPi.GPIO as GPIO  # Raspberry Pi GPIO 控制
import time

# # 設定 GPIO
# VIBRATION_PIN =   # 設定 GPIO 腳位（可根據實際連接更改）
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(VIBRATION_PIN, GPIO.OUT)

# def activate_vibration(duration=1):
#     GPIO.output(VIBRATION_PIN, GPIO.HIGH)
#     time.sleep(duration)
#     GPIO.output(VIBRATION_PIN, GPIO.LOW)

# 初始化 pyttsx3 引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 調整語速
engine.setProperty('volume', 1.0)  # 調整音量

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
model.load_state_dict(torch.load('modle/complete_deep_cnn_lstm.pth', map_location=torch.device('cpu')))
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
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and sequence dimension
        lengths = torch.tensor([1], dtype=torch.int64).cpu()  # Length is 1 for a single frame
        outputs = model(image_tensor, lengths)
        predicted = torch.sigmoid(outputs).item()
    return predicted

for i in range(12):
    randint_ = [27, 118, 140, 14, 3, 32, 150, 155, 81, 54, 160, 89, 126]  # 隨機選取12張圖片，可以更改
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
                engine.say("請減速慢行")
                engine.runAndWait()
                # activate_vibration(2)  # 震動 2 秒
        
        accident_prediction = "請減速慢行" if tf else "安全"
        cv2.putText(frame, accident_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if accident_prediction == "安全" else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("image", frame)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

# # 清理 GPIO
# GPIO.cleanup()


import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
import os
from random import randint
import pyttsx3  # 確保安裝了 pyttsx3
# import RPi.GPIO as GPIO  # Raspberry Pi GPIO 控制
import time
from PIL import Image, ImageDraw, ImageFont

# # 設定 GPIO
# VIBRATION_PIN = 18  # 設定 GPIO 腳位（可根據實際連接更改）
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(VIBRATION_PIN, GPIO.OUT)

# def activate_vibration(duration=1):
#     GPIO.output(VIBRATION_PIN, GPIO.HIGH)
#     time.sleep(duration)
#     GPIO.output(VIBRATION_PIN, GPIO.LOW)

# 初始化 pyttsx3 引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 調整語速
engine.setProperty('volume', 1.0)  # 調整音量

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
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and sequence dimension
        lengths = torch.tensor([1], dtype=torch.int64).cpu()  # Length is 1 for a single frame
        outputs = model(image_tensor, lengths)
        predicted = torch.sigmoid(outputs).item()
    return predicted

def draw_text_chinese(frame, text, position, font_path="C:/Windows/Fonts/msjh.ttc", font_size=60):
    """
    使用 PIL 在 OpenCV 圖片上繪製中文字
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 轉換為 PIL 格式
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)  # 指定字體檔案與大小
    except OSError:
        print(f"字體檔案無法找到: {font_path}")
        return frame  # 如果找不到字體，就返回原圖
    color = (255, 0, 0) if text == "請減速慢行" else (0, 255, 0)  # 紅色: 請減速慢行, 綠色: 安全
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # 轉回 OpenCV 格式


for i in range(12):
    randint_ = [27, 118, 140, 14, 3, 32, 150, 155, 81, 54, 160, 89, 126]  # 隨機選取12張圖片，可以更改
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
                engine.say("請減速慢行")
                engine.runAndWait()
                # activate_vibration(2)  # 震動 2 秒
        
        accident_prediction = "請減速慢行" if tf else "安全"
        frame = draw_text_chinese(frame, accident_prediction, (10, 30))
        cv2.imshow("image", frame)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

# # 清理 GPIO
# GPIO.cleanup()



# 2025-02-20 改良

import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
import os
from random import randint
import pyttsx3
import time
from PIL import Image, ImageDraw, ImageFont
import threading  # 新增多執行緒
import random

# 初始化 pyttsx3 引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 調整語速
engine.setProperty('volume', 1.0)  # 調整音量

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
    使用 PIL 在 OpenCV 圖片上繪製中文字
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

# 建立語音播報的執行緒函式
def play_audio(message):
    engine.say(message)
    engine.runAndWait()

# 處理影像的主迴圈
for i in range(12):
    # randint_ = [27, 118, 140, 14, 3, 32, 150, 155, 81, 54, 160, 89, 126]
    randint_ = random.sample(range(0, 179), 12)  # 隨機選取 12 張圖片
    print(randint_)
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
                audio_thread = threading.Thread(target=play_audio, args=("請減速慢行",))
                audio_thread.start()  # 啟動語音播報執行緒

        accident_prediction = "請減速慢行" if tf else "安全"
        frame = draw_text_chinese(frame, accident_prediction, (10, 30))
        cv2.imshow("image", frame)
        cv2.waitKey(50)

    cv2.destroyAllWindows()
    
    
 #--------------------------------------------------------------------------------  
    
    
import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
import os
import pyttsx3
import time
from PIL import Image, ImageDraw, ImageFont
import threading  # 使用多執行緒確保同步
import random

# 初始化 pyttsx3 引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 調整語速
engine.setProperty('volume', 1.0)  # 調整音量

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
    使用 PIL 在 OpenCV 圖片上繪製中文字
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

# 建立語音與影像同步執行函式
def play_audio_and_display_text(frame, message, position):
    """
    使用多執行緒確保語音與影像顯示同步
    """
    audio_thread = threading.Thread(target=engine.say, args=(message,))
    audio_thread.start()  # 啟動語音播放
    frame = draw_text_chinese(frame, message, position)  # 繪製文字
    return frame  # 回傳處理後的影像

# 處理影像的主迴圈
for i in range(12):
    # randint_ = [27, 118, 140, 14, 3, 32, 150, 155, 81, 54, 160, 89, 126]
    randint_ = random.sample(range(0, 179), 12)  # 隨機選取 12 張圖片
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
                frame = play_audio_and_display_text(frame, "請減速慢行", (10, 30))

        accident_prediction = "請減速慢行" if tf else "安全"
        frame = draw_text_chinese(frame, accident_prediction, (10, 30))  # 繪製文字
        cv2.imshow("image", frame)
        cv2.waitKey(50)

    cv2.destroyAllWindows()







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
