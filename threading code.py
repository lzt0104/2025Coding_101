import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
import os
import time
from PIL import Image, ImageDraw, ImageFont
import random
import threading
from queue import Queue
from pygame import mixer  # 用於播放音檔

# 初始化 pygame.mixer 用於音檔播放
mixer.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定義模型
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
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
        dropout = 0 if num_layers == 1 else 0.5
        self.lstm = nn.LSTM(1280, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
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

# 用於執行緒間通訊的佇列
image_queue = Queue()
prediction_queue = Queue()

# 用於同步音檔播放的事件
sound_playing = threading.Event()
sound_playing.set()  # 初始設為已完成

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

def play_sound(file_path="slow_down.mp3"):
    """播放音檔並等待完成"""
    global sound_playing
    sound_playing.clear()  # 標記音檔播放中
    sound = mixer.Sound(file_path)
    sound.play()
    while mixer.get_busy():  # 等待音檔播放完成
        time.sleep(0.1)
    sound_playing.set()  # 標記音檔播放完成

# 執行緒工作函數
def image_loader_worker(folder):
    images = [img for img in os.listdir(folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('.')[0]))
    for image in images:
        img_path = os.path.join(folder, image)
        frame = cv2.imread(img_path)
        image_tensor = preprocess_image(img_path)
        image_queue.put((frame, image_tensor))
    image_queue.put((None, None))  # 結束信號

def prediction_and_display_worker():
    tf = 0
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    while True:
        frame, image_tensor = image_queue.get()
        if frame is None:  # 結束信號
            break
        
        # 預測
        if not tf:
            prediction = predict_single_frame(model, image_tensor)
            if prediction > 0.5:
                tf = 1
                threading.Thread(target=play_sound, args=("slow_down.mp3",), daemon=True).start()
        
        # 顯示
        accident_prediction = "請減速慢行" if tf else "安全"
        frame = draw_text_chinese(frame, accident_prediction, (10, 30))
        cv2.imshow("image", frame)
        cv2.waitKey(50)
        image_queue.task_done()
    
    cv2.destroyAllWindows()
    prediction_queue.put(tf)  # 將危險狀態傳遞給主線程

# 主處理函數
def process_folder(folder):
    loader_thread = threading.Thread(target=image_loader_worker, args=(folder,), daemon=True)
    pred_display_thread = threading.Thread(target=prediction_and_display_worker, daemon=True)

    loader_thread.start()
    pred_display_thread.start()

    loader_thread.join()
    pred_display_thread.join()
    
    # 等待音檔播放完成（如果有）
    sound_playing.wait()
    return prediction_queue.get()  # 獲取危險狀態

# 主迴圈
for i in range(12):
    randint_ = random.sample(range(0, 179), 12)
    image_folder = f"train/train/freeway_{randint_[i]:04d}"
    process_folder(image_folder)

# 清理
mixer.quit()
cv2.destroyAllWindows()

