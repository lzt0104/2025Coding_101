import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import pyttsx3
import time
import threading

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 初始化語音引擎和控制變數
tts_engine = None
tts_thread = None
is_speaking = False
stop_speaking = False

# 初始化語音引擎
def init_tts():
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150) 
        tts_engine.setProperty('volume', 1.0)
        return True
    except Exception as e:
        print(f"語音引擎初始化失敗: {e}")
        return False

# 語音播放線程函數
def speak_thread_func(text):
    global is_speaking, stop_speaking
    try:
        is_speaking = True
        stop_speaking = False
        
        tts_engine.say(text)
        tts_engine.startLoop(False)
        
        while not stop_speaking and tts_engine.isBusy():
            tts_engine.iterate()
            time.sleep(0.1)
            
        tts_engine.endLoop()
    except Exception as e:
        print(f"語音播報錯誤: {e}")
    finally:
        is_speaking = False

# 開始語音警告
def start_warning():
    global tts_thread, is_speaking, tts_engine
    
    if tts_engine is None:
        if not init_tts():
            return
    
    if is_speaking:
        return
        
    tts_thread = threading.Thread(target=speak_thread_func, args=("Slow down",))
    tts_thread.daemon = True
    tts_thread.start()

# 停止語音警告
def stop_warning():
    global stop_speaking
    stop_speaking = True

# 定義模型
class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetLSTM, self).__init__()
        from torchvision.models import resnet18
        self.feature_extractor = resnet18(weights=None)
        self.feature_size = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        features = [self.feature_extractor(x[:, i]) for i in range(seq_len)]
        x = torch.stack(features, dim=1)
        lstm_out, _ = self.lstm(x)
        output = self.classifier(lstm_out[:, -1, :])
        return output

# 安全預警演示類
class SafetyDemo:
    def __init__(self, model_path='traffic_model_state_dict.pth', sequence_length=5):
        self.sequence_length = sequence_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = self.load_model(model_path)
        self.prediction_buffer = []
        self.buffer_size = 3
        self.previous_risk_state = False
    
    def load_model(self, model_path):
        model = ResNetLSTM().to(device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            print("成功載入模型")
        else:
            print("未找到模型，使用未訓練模型")
        model.eval()
        return model
    
    def process_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        return self.transform(pil_image)
    
    def predict(self, image_tensors):
        if len(image_tensors) > self.sequence_length:
            image_tensors = image_tensors[-self.sequence_length:]
        elif len(image_tensors) < self.sequence_length:
            image_tensors += [image_tensors[-1]] * (self.sequence_length - len(image_tensors))
        
        tensor_sequence = torch.stack(image_tensors).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(tensor_sequence)
            probability = torch.sigmoid(output).item()
        return probability
    
    def process_camera(self):
        # 使用 OpenCV 直接訪問相機
        print("正在初始化相機...")
        cap = cv2.VideoCapture(0)  # 打開默認相機
        
        if not cap.isOpened():
            print("無法開啟相機。嘗試其他相機索引：")
            # 嘗試其他幾個可能的相機索引
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"成功打開相機索引 {i}")
                    break
            
            if not cap.isOpened():
                print("無法打開相機。請檢查相機連接或驅動。")
                return
        
        # 設置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("相機初始化完成，開始處理視頻流")
        tensor_buffer = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("無法獲取影像幀")
                    break
                
                tensor = self.process_image(frame)
                tensor_buf-fer.append(tensor)
                if len(tensor_buffer) > self.sequence_length:
                    tensor_buffer.pop(0)
                
                if len(tensor_buffer) == self.sequence_length:
                    probability = self.predict(tensor_buffer)
                    is_risky = probability > 0.5
                    
                    if is_risky and not self.previous_risk_state:
                        start_warning()
                    elif not is_risky and self.previous_risk_state:
                        stop_warning()
                    
                    self.previous_risk_state = is_risky
                    
                    # 顯示結果在畫面上
                    color = (0, 0, 255) if is_risky else (0, 255, 0)
                    text = "SLOW DOWN" if is_risky else "SAFE"
                    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(frame, f"Risk: {probability:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    cv2.imshow("Traffic Safety Alert", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("程式被使用者中斷")
        except Exception as e:
            print(f"處理視頻時出錯: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("相機已關閉")

# 主程式
def main():
    print("=== 交通安全預警系統 ===")
    init_tts()
    demo = SafetyDemo(sequence_length=5)
    demo.process_camera()
    stop_warning()
    print("程式結束")

if __name__ == "__main__":
    main()