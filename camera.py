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
# 使用 picamera2 套件取代舊版 picamera
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
# 加入 GPIO 控制，用於振動馬達和蜂鳴器
import RPi.GPIO as GPIO

# 設置 PyTorch 運算設備 (優先使用 GPU 加速)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# ===== GPIO 設置區域 =====
# 使用 BCM 編號方式
GPIO.setmode(GPIO.BCM)
# 定義振動馬達和蜂鳴器的 GPIO 引腳
VIBRATION_MOTOR_PIN = 17  # 振動馬達連接的 GPIO 引腳
BUZZER_PIN = 18          # 蜂鳴器連接的 GPIO 引腳
# 設置為輸出模式
GPIO.setup(VIBRATION_MOTOR_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
# 初始化 PWM 控制
vibration_pwm = GPIO.PWM(VIBRATION_MOTOR_PIN, 100)  # 振動馬達 PWM 頻率為 100Hz
buzzer_pwm = GPIO.PWM(BUZZER_PIN, 440)           # 蜂鳴器頻率為 440Hz (A4 音符)

# ===== 語音引擎相關變數 =====
tts_engine = None       # 語音引擎物件
tts_thread = None       # 語音執行緒
is_speaking = False     # 是否正在播放語音
stop_speaking = False   # 是否應該停止語音
is_warning = False      # 是否正在發出警告 (振動+蜂鳴器+語音)

# ===== 語音引擎功能 =====
# 初始化語音引擎
def init_tts():
    """初始化文字轉語音引擎"""
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)    # 語速設置
        tts_engine.setProperty('volume', 1.0)  # 音量設置
        return True
    except Exception as e:
        print(f"語音引擎初始化失敗: {e}")
        return False

# 語音播放線程函數
def speak_thread_func(text):
    """
    在獨立執行緒中執行語音播放，避免阻塞主程序
    
    參數:
    text (str): 要播放的語音文字
    """
    global is_speaking, stop_speaking
    try:
        is_speaking = True
        stop_speaking = False
        
        tts_engine.say(text)
        tts_engine.startLoop(False)
        
        # 持續檢查是否需要停止語音
        while not stop_speaking and tts_engine.isBusy():
            tts_engine.iterate()
            time.sleep(0.1)
            
        tts_engine.endLoop()
    except Exception as e:
        print(f"語音播報錯誤: {e}")
    finally:
        is_speaking = False

# ===== 警告控制功能 =====
# 開始警告 (語音、振動和蜂鳴器)
def start_warning():
    """
    啟動多重警告系統：振動馬達、蜂鳴器和語音提示
    同時被觸發以提供最佳警告效果
    """
    global tts_thread, is_speaking, tts_engine, is_warning
    
    # 避免重複觸發警告機制
    if is_warning:
        return
        
    is_warning = True
    
    # 啟動振動馬達 - 強度適中以確保明顯感知但不干擾操作
    vibration_pwm.start(70)  # PWM 佔空比 70%
    
    # 啟動蜂鳴器 - 清晰可聽但不過於刺耳
    buzzer_pwm.start(50)  # PWM 佔空比 50%
    
    # 啟動語音提示
    if tts_engine is None:
        if not init_tts():
            return
    
    if not is_speaking:
        tts_thread = threading.Thread(target=speak_thread_func, args=("Slow down",))
        tts_thread.daemon = True  # 設為守護執行緒，主程式結束時會自動終止
        tts_thread.start()

# 停止警告
def stop_warning():
    """
    停止所有警告系統：振動馬達、蜂鳴器和語音提示
    """
    global stop_speaking, is_warning
    
    if not is_warning:
        return
        
    is_warning = False
    
    # 停止振動馬達
    vibration_pwm.ChangeDutyCycle(0)
    
    # 停止蜂鳴器
    buzzer_pwm.ChangeDutyCycle(0)
    
    # 停止語音提示
    stop_speaking = True

# ===== 深度學習模型定義 =====
class ResNetLSTM(nn.Module):
    """
    結合 ResNet 和 LSTM 的深度學習模型
    用於序列影像的危險情境辨識
    
    ResNet 用於單幀特徵提取
    LSTM 用於學習時間序列特徵關係
    """
    def __init__(self, num_classes=1):
        super(ResNetLSTM, self).__init__()
        from torchvision.models import resnet18
        # 使用 ResNet18 作為特徵提取器
        self.feature_extractor = resnet18(weights=None)
        self.feature_size = self.feature_extractor.fc.in_features
        # 移除最後的全連接層，只保留特徵提取部分
        self.feature_extractor.fc = nn.Identity()
        
        # 雙向 LSTM 用於序列處理
        self.lstm = nn.LSTM(
            input_size=512,           # ResNet18 特徵維度
            hidden_size=256,          # LSTM 隱藏層大小
            num_layers=2,             # LSTM 層數
            batch_first=True,         # 批次維度在前
            dropout=0.5,              # 避免過擬合
            bidirectional=True        # 雙向 LSTM 增強學習能力
        )
        
        # 分類器網路
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),      # 512=雙向LSTM的輸出維度 (256*2)
            nn.ReLU(inplace=True),    # 使用 ReLU 激活函數
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        前向傳播函數
        
        參數:
        x: 形狀為 [batch_size, seq_len, channels, height, width] 的輸入張量
        
        返回:
        模型的輸出張量
        """
        batch_size, seq_len, C, H, W = x.size()
        # 對序列中的每一幀提取特徵
        features = [self.feature_extractor(x[:, i]) for i in range(seq_len)]
        # 將特徵堆疊成序列
        x = torch.stack(features, dim=1)
        # 通過 LSTM 處理特徵序列
        lstm_out, _ = self.lstm(x)
        # 只取最後時間步的輸出進行分類
        output = self.classifier(lstm_out[:, -1, :])
        return output

# ===== 安全預警系統主類 =====
class SafetyDemo:
    """
    交通安全預警系統主類
    負責模型載入、影像處理和風險預測
    """
    def __init__(self, model_path='traffic_model_state_dict.pth', sequence_length=5):
        """
        初始化安全預警系統
        
        參數:
        model_path (str): 模型權重檔案路徑
        sequence_length (int): 影像序列長度，影響時間相關性判斷
        """
        self.sequence_length = sequence_length
        # 影像預處理轉換
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),     # 調整圖片大小為 ResNet 的輸入尺寸
            transforms.ToTensor(),             # 轉換為張量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 標準化
        ])
        # 載入模型
        self.model = self.load_model(model_path)
        # 預測結果緩衝區，用於平滑化預測結果
        self.prediction_buffer = []
        self.buffer_size = 3
        # 記錄前一個風險狀態，用於檢測狀態變化
        self.previous_risk_state = False
    
    def load_model(self, model_path):
        """
        載入預訓練模型
        
        參數:
        model_path (str): 模型權重檔案路徑
        
        返回:
        載入權重的模型
        """
        model = ResNetLSTM().to(device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            print("成功載入模型")
        else:
            print("未找到模型，使用未訓練模型")
        model.eval()  # 設置為評估模式
        return model
    
    def process_image(self, image):
        """
        處理輸入的 OpenCV 影像以適合模型輸入
        
        參數:
        image: OpenCV 格式的影像 (BGR)
        
        返回:
        經過處理的影像張量
        """
        # 將 BGR 轉換為 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 轉換為 PIL 格式
        pil_image = Image.fromarray(image_rgb)
        # 應用預處理轉換
        return self.transform(pil_image)
    
    def predict(self, image_tensors):
        """
        預測影像序列的風險概率
        
        參數:
        image_tensors: 影像張量列表
        
        返回:
        風險概率 (0-1 之間)
        """
        # 確保序列長度正確
        if len(image_tensors) > self.sequence_length:
            # 如果序列過長，只保留最後 sequence_length 幀
            image_tensors = image_tensors[-self.sequence_length:]
        elif len(image_tensors) < self.sequence_length:
            # 如果序列過短，使用最後一幀填充
            image_tensors += [image_tensors[-1]] * (self.sequence_length - len(image_tensors))
        
        # 準備模型輸入
        tensor_sequence = torch.stack(image_tensors).unsqueeze(0).to(device)
        # 執行推論
        with torch.no_grad():
            output = self.model(tensor_sequence)
            # 將輸出轉換為概率
            probability = torch.sigmoid(output).item()
        return probability
    
    def process_camera(self):
        """
        處理相機串流，進行實時風險預測和警告
        """
        # 初始化 picamera2
        picam2 = Picamera2()
        # 配置相機參數
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        # 啟動相機
        picam2.start()
        time.sleep(2)  # 給相機啟動時間
            
        # 影像張量緩衝區
        tensor_buffer = []
        try:
            while True:
                # 擷取一幀影像
                image = picam2.capture_array()
                
                # 處理影像並添加到緩衝區
                tensor = self.process_image(image)
                tensor_buffer.append(tensor)
                # 維持緩衝區大小
                if len(tensor_buffer) > self.sequence_length:
                    tensor_buffer.pop(0)
                
                # 當累積足夠的幀數時進行預測
                if len(tensor_buffer) == self.sequence_length:
                    # 獲取風險概率
                    probability = self.predict(tensor_buffer)
                    # 判斷是否處於風險狀態 (概率 > 0.5)
                    is_risky = probability > 0.5
                    
                    # 根據風險狀態變化控制警告
                    if is_risky and not self.previous_risk_state:
                        # 風險狀態開始，啟動警告
                        start_warning()
                    elif not is_risky and self.previous_risk_state:
                        # 風險狀態結束，停止警告
                        stop_warning()
                    
                    # 更新前一個風險狀態
                    self.previous_risk_state = is_risky
                    
                    # 在影像上顯示狀態
                    color = (0, 0, 255) if is_risky else (0, 255, 0)  # 危險為紅色，安全為綠色
                    cv2.putText(image, "SLOW DOWN" if is_risky else "SAFE", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    # 顯示影像
                    cv2.imshow("Traffic Safety Alert", image)
                    # 檢查退出按鍵 (q)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except KeyboardInterrupt:
            print("程式被使用者中斷")
        finally:
            # 確保所有資源都被正確釋放
            picam2.stop()
            cv2.destroyAllWindows()
            stop_warning()
            # 清理 GPIO 資源
            vibration_pwm.stop()
            buzzer_pwm.stop()
            GPIO.cleanup()
            print("相機和 GPIO 已關閉")

# ===== 主程式 =====
def main():
    """主程式入口點"""
    print("=== 交通安全預警系統 ===")
    # 初始化語音引擎
    init_tts()
    # 創建並啟動安全預警系統
    demo = SafetyDemo(sequence_length=5)
    demo.process_camera()
    print("程式結束")

# 程式入口點
if __name__ == "__main__":
    main()