import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import pyttsx3
import time
import random
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
        
        # 語音播放
        tts_engine.say(text)
        tts_engine.startLoop(False)
        
        # 持續檢查是否應該停止
        while not stop_speaking and tts_engine.isBusy():
            tts_engine.iterate()
            time.sleep(0.1)
            
        # 確保停止
        tts_engine.endLoop()
    except Exception as e:
        print(f"語音播報錯誤: {e}")
    finally:
        is_speaking = False

# 開始語音警告
def start_warning():
    global tts_thread, is_speaking, tts_engine
    
    # 確保引擎已初始化
    if tts_engine is None:
        if not init_tts():
            return
    
    # 如果已經在播放，不要重複啟動
    if is_speaking:
        return
        
    # 啟動語音警告線程
    tts_thread = threading.Thread(target=speak_thread_func, args=("Slow down",))
    tts_thread.daemon = True
    tts_thread.start()

# 停止語音警告
def stop_warning():
    global stop_speaking
    stop_speaking = True

# 定義與原始模型相容的ResNet+LSTM模型
class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetLSTM, self).__init__()
        
        # 使用預訓練的 ResNet18 作為特徵提取器
        from torchvision.models import resnet18, ResNet18_Weights
        self.feature_extractor = resnet18(weights=None)  # 不使用預訓練權重
        self.feature_size = self.feature_extractor.fc.in_features  # 通常是 512
        self.feature_extractor.fc = nn.Identity()  # 移除分類層
        
        # 注意力機制 - 與原始模型的注意力層匹配
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # LSTM 層 - 與原始模型匹配
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        # 分類器 - 與原始模型匹配
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # 處理每一幀
        features = []
        for i in range(seq_len):
            frame = x[:, i]
            frame_feature = self.feature_extractor(frame)
            features.append(frame_feature)
        
        # 合併特徵
        x = torch.stack(features, dim=1)  # [batch_size, seq_len, feature_size]
        
        # LSTM處理
        lstm_out, _ = self.lstm(x)
        
        # 應用注意力機制
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 分類
        output = self.classifier(context_vector)
        
        return output

# 安全預警演示類
class SafetyDemo:
    def __init__(self, model_path='model/traffic_model_state_dict.pth', sequence_length=5):
        self.sequence_length = sequence_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 載入模型
        self.model = self.load_model(model_path)
        
        # 設置顯示參數
        self.safe_color = (0, 255, 0)  # 綠色
        self.warning_color = (0, 0, 255)  # 紅色
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 平滑預測 - 使用最近幾幀的平均來減少誤報
        self.prediction_buffer = []
        self.buffer_size = 3
        
        # 狀態追蹤
        self.previous_risk_state = False
    
    def load_model(self, model_path):
        """載入或創建模型"""
        # 創建模型實例
        model = ResNetLSTM().to(device)
        
        # 嘗試不同的模型路徑
        model_paths = [
            model_path,
            'model/traffic_model_state_dict.pth',
            os.path.join(os.getcwd(), 'model/traffic_model_state_dict.pth')
        ]
        
        # 輸出當前工作目錄，幫助診斷
        print(f"當前工作目錄: {os.getcwd()}")
        
        # 檢查模型檔案是否存在
        for path in model_paths:
            if os.path.exists(path):
                try:
                    print(f"嘗試載入模型: {path}")
                    
                    # 加載模型
                    checkpoint = torch.load(path, map_location=device)
                    
                    # 檢查加載的對象類型
                    if isinstance(checkpoint, dict):
                        # 如果是狀態字典或檢查點
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        else:
                            # 直接載入狀態字典，允許不完全匹配
                            model.load_state_dict(checkpoint, strict=False)
                        print(f"成功載入模型參數: {path}")
                        break
                    else:
                        # 如果是完整模型
                        print("載入的是完整模型，使用該模型")
                        model = checkpoint.to(device)
                        break
                except Exception as e:
                    print(f"載入模型 {path} 出錯: {e}")
            else:
                print(f"模型文件不存在: {path}")
        else:
            print("所有模型路徑都無效，使用未訓練的模型")
        
        model.eval()
        return model
    
    def process_image(self, image):
        """處理單張圖像"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        return self.transform(pil_image)
    
    def predict(self, image_tensors):
        """預測序列中的風險"""
        if len(image_tensors) != self.sequence_length:
            # 調整序列長度
            if len(image_tensors) > self.sequence_length:
                image_tensors = image_tensors[-self.sequence_length:]
            else:
                image_tensors = image_tensors + [image_tensors[-1]] * (self.sequence_length - len(image_tensors))
        
        # 將圖像堆疊為張量
        tensor_sequence = torch.stack(image_tensors).unsqueeze(0).to(device)
        
        # 預測
        with torch.no_grad():
            try:
                output = self.model(tensor_sequence)
                probability = torch.sigmoid(output).item()
                return probability
            except Exception as e:
                print(f"預測錯誤: {e}")
                return 0.0
    
    def process_folder(self, folder_path):
        """處理資料夾"""
        # 檢查資料夾是否存在
        if not os.path.exists(folder_path):
            print(f"資料夾不存在: {folder_path}")
            return False
        
        # 獲取所有圖像
        images = sorted([
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ], key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
        
        if len(images) < self.sequence_length:
            print(f"圖像數量不足: {len(images)}")
            return False
        
        print(f"處理資料夾: {folder_path} (共 {len(images)} 張圖像)")
        
        # 創建窗口
        cv2.namedWindow("Traffic Safety Alert", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Traffic Safety Alert", 800, 600)
        
        # 初始化緩衝區
        tensor_buffer = []
        self.prediction_buffer = []
        self.previous_risk_state = False
        risk_detected = False
        
        try:
            # 處理每張圖像
            for img_name in images:
                img_path = os.path.join(folder_path, img_name)
                frame = cv2.imread(img_path)
                
                if frame is None:
                    continue
                
                # 處理圖像並加入緩衝區
                tensor = self.process_image(frame)
                tensor_buffer.append(tensor)
                
                # 保持緩衝區大小
                if len(tensor_buffer) > self.sequence_length:
                    tensor_buffer.pop(0)
                
                # 進行預測
                if len(tensor_buffer) == self.sequence_length:
                    probability = self.predict(tensor_buffer)
                    
                    # 更新預測緩衝區進行平滑化
                    self.prediction_buffer.append(probability > 0.5)
                    if len(self.prediction_buffer) > self.buffer_size:
                        self.prediction_buffer.pop(0)
                    
                    # 當至少有一半的預測為風險時認定為風險狀態
                    votes = sum(self.prediction_buffer)
                    current_risk = votes >= (len(self.prediction_buffer) / 2)
                    
                    # 檢查風險狀態是否改變
                    if current_risk and not self.previous_risk_state:
                        # 從安全變為風險 - 啟動警告
                        start_warning()
                    elif not current_risk and self.previous_risk_state:
                        # 從風險變為安全 - 停止警告
                        stop_warning()
                    
                    # 更新狀態
                    self.previous_risk_state = current_risk
                    risk_detected = risk_detected or current_risk
                    
                    # 在圖像上添加顯示
                    display_frame = frame.copy()
                    
                    # 只在檢測到風險時顯示警告，否則顯示安全
                    status_text = "SLOW DOWN" if current_risk else "SAFE"
                    color = self.warning_color if current_risk else self.safe_color
                    
                    # 顯示狀態
                    cv2.putText(
                        display_frame, status_text, (50, 50), 
                        self.font, 1.0, color, 2
                    )
                    
                    # 顯示概率
                    cv2.putText(
                        display_frame, f"Risk: {probability:.2f}", (50, 100), 
                        self.font, 0.8, (255, 255, 255), 2
                    )
                    
                    # 顯示圖像
                    cv2.imshow("Traffic Safety Alert", display_frame)
                    key = cv2.waitKey(50)  # 50ms顯示每幀
                    
                    if key == 27 or key == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("用戶中斷處理")
        finally:
            # 確保關閉語音和窗口
            stop_warning()
            cv2.destroyAllWindows()
        
        return risk_detected
    
    def run_demo(self, base_path="train/train", num_folders=3):
        """運行演示"""
        # 檢查路徑
        if not os.path.exists(base_path):
            potential_paths = ["train/train", "train", "."]
            for path in potential_paths:
                if os.path.exists(path):
                    base_path = path
                    break
            else:
                print("找不到測試資料夾")
                return
        
        # 找到所有資料夾
        folders = [
            f for f in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, f)) and f.startswith('freeway_')
        ]
        
        if not folders:
            print(f"在 {base_path} 中找不到測試資料夾")
            return
        
        # 隨機選擇資料夾
        if len(folders) > num_folders:
            folders = random.sample(folders, num_folders)
        
        results = []
        
        # 處理每個資料夾
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            print(f"\n分析影片: {folder}")
            
            try:
                risk_detected = self.process_folder(folder_path)
                results.append({'folder': folder, 'risk_detected': risk_detected})
            except Exception as e:
                print(f"處理 {folder} 時出錯: {e}")
        
        # 顯示結果摘要
        if results:
            print("\n===== 分析結果 =====")
            for result in results:
                status = "危險" if result['risk_detected'] else "安全"
                print(f"{result['folder']}: {status}")

# 主程式
def main():
    print("=== 交通安全預警系統演示 ===")
    
    # 初始化語音引擎
    init_tts()
    
    try:
        # 創建演示實例
        demo = SafetyDemo(sequence_length=5)
        
        # 運行演示
        demo.run_demo(num_folders=3)
    
    except Exception as e:
        print(f"程式執行錯誤: {e}")
    finally:
        # 確保停止語音和關閉窗口
        stop_warning()
        cv2.destroyAllWindows()
        print("程式結束")

if __name__ == "__main__":
    main()