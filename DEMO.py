import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import os
from random import randint
import pandas as pd
import pyttsx3  # 確保安裝了 pyttsx3：pip install pyttsx3

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
model.load_state_dict(torch.load('model/modle.pth', map_location=torch.device('cpu')))  #更換路徑
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
    randint_ = [27,118,140,14,3,32,150,155,81,54,160,89,126] #隨機選取12張圖片，可以更改
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
            if prediction>0.5:
                tf =  1
        accident_prediction = "Please reduce the speed" if tf  else "safe"
        
        cv2.putText(frame, accident_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if accident_prediction == "safe" else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("image", frame)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

改良一
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
        accident_prediction = "Please reduce the speed" if tf else "safe"

        # 加入語音提示
        if accident_prediction == "Please reduce the speed":
            engine.say("請減速慢行")
            engine.runAndWait()

        cv2.putText(frame, accident_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if accident_prediction == "safe" else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("image", frame)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

# #改良二
# # 設定影片參數
# frame_width, frame_height = 1280, 720  # 設定影片解析度
# fps = 20  # 每秒影格數
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 指定影片格式
# output_video_path = "output_video.avi"  # 輸出的影片檔案名

# # 初始化 VideoWriter
# video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# randint_ = [27, 118, 140, 14, 3, 32, 150, 155, 81, 54, 160, 89, 126]  # 隨機選取12個目錄

# for i in range(12):
#     image_folder = f"train/train/freeway_{randint_[i]:04d}"

#     images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#     images.sort(key=lambda x: int(x.split('.')[0]))

#     tf = 0

#     for image in images:
#         img_path = os.path.join(image_folder, image)
#         frame = cv2.imread(img_path)

#         # 調整圖片大小與影片解析度匹配
#         frame = cv2.resize(frame, (frame_width, frame_height))

#         image_tensor = preprocess_image(img_path)
#         if not tf:
#             prediction = predict_single_frame(model, image_tensor)
#             if prediction > 0.5:
#                 tf = 1
#         accident_prediction = "Please reduce the speed" if tf else "safe"

#         # 加入語音提示
#         if accident_prediction == "Please reduce the speed":
#             engine.say("請減速慢行")
#             engine.runAndWait()

#         # 在影像上顯示提示文字
#         cv2.putText(
#             frame,
#             accident_prediction,
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0) if accident_prediction == "safe" else (0, 0, 255),
#             2,
#             cv2.LINE_AA
#         )

#         # 將處理後的影像幀寫入影片
#         video_writer.write(frame)

# # 釋放資源
# video_writer.release()
# cv2.destroyAllWindows()

# print(f"影片已保存至 {output_video_path}")



# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from matplotlib import rcParams

# # 設定中文字體
# rcParams['font.family'] = 'HarmonyOS Sans TC'  # 更換為系統中已安裝的中文字體名稱
# rcParams['axes.unicode_minus'] = False  # 確保負號正常顯示

# # 混淆矩陣數據
# confusion_matrix = np.array([[956, 8], [29, 134]])
# labels = ["T", "F"]

# # 繪製混淆矩陣
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False,annot_kws={"size": 14})

# # 標題與座標
# plt.title("Confusion Matrix", fontsize=16)
# plt.xlabel("Predicted Label", fontsize=14)
# plt.ylabel("True Label", fontsize=14)

# # 顯示圖表
# plt.tight_layout()
# plt.show()

# from sklearn.metrics import classification_report
# import numpy as np

# # 模擬測試數據
# y_true = np.array([0] * 956 + [1] * 8 + [0] * 29 + [1] * 134)  # 真實標籤
# y_pred = np.array([0] * 956 + [0] * 8 + [1] * 29 + [1] * 134)  # 預測標籤

# # 打印分類報告
# print(classification_report(y_true, y_pred, target_names=["T", "F"]))



