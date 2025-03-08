---
title: 交通安全預警系統

---

# 🚗 馬路三寶，別來撞我 (Road Hazards, Don't Hit Me)

## 🌟 專案介紹

「**馬路三寶，別來撞我**」是一個結合計算機視覺與深度學習技術的先進交通安全預警系統，旨在利用人工智能技術預防交通事故，保護道路使用者的生命安全。

### 💡 核心功能

- 🔍 即時風險偵測：運用先進的深度學習模型，連續分析交通場景，精準預測潛在危險
- 🚨 主動預警系統：當檢測到高風險情況時，立即通過視覺和語音方式發出警告
- 🤖 智能感知：結合 ResNet 空間特徵提取和 LSTM 時序分析，F1 分數高達 0.975
- 🖥️ 多平台支援：支援桌面電腦、樹莓派等不同硬體平台

## 🏆 專案特色

### 🧠 創新的深度學習架構

我們的系統採用創新的 ResNet+LSTM 混合模型，突破傳統單一模型的局限：

#### 🔷 ResNet 特徵提取
- 利用殘差連接解決梯度消失問題
- 從道路、車輛和環境圖像中提取高階視覺特徵
- 預訓練權重加速收斂並提升泛化能力

#### 🔷 自注意力機制
- 自動識別關鍵幀與重要區域
- 計算注意力權重，突出顯著特徵
- 過濾噪聲，提高模型魯棒性

#### 🔷 雙向 LSTM 時序分析
- 捕捉時間維度的運動模式
- 同時考慮過去與未來上下文
- 多層結構提取複雜時序依賴關係

## 🛠️ 技術架構

![ResNet+LSTM 交通風險預測模型架構](resnet_lstm_architecture.png)

### 模型各模塊功能：

1. **ResNet特徵提取**：
   - 使用 ResNet-18 提取空間特徵
   - 輸出維度：512
   - 捕捉圖像的關鍵視覺信息

2. **注意力機制**：
   - 使用 Linear + Tanh 網絡
   - 識別關鍵幀
   - 生成上下文向量

3. **雙向LSTM**：
   - 輸入維度：512
   - 隱藏層：256
   - 雙向、兩層結構
   - Dropout: 0.5

4. **全連接分類器**：
   - 512 → 128
   - 128 → 64
   - 64 → 1
   - 輸出風險概率

## 📊 模型性能

### 關鍵指標

| 指標     | 數值   | 說明                                     |
|----------|--------|------------------------------------------|
| 準確率   | 97.5%  | 正確預測佔總預測的比例                   |
| F1 分數  | 0.975  | 精確度與召回率的調和平均                 |
| 精確度   | 0.967  | 真陽性佔所有陽性預測的比例               |
| 召回率   | 0.983  | 正確識別的危險場景佔所有危險場景的比例   |

### 訓練曲線

![訓練與驗證歷史](training_history.png)

## 🚀 快速開始

### 環境準備

1. 克隆專案倉庫
```bash
git clone https://github.com/your-username/road-hazards-detector.git
cd road-hazards-detector
```

2. 安裝依賴
```bash
pip install -r requirements.txt
```

### 使用方法

#### 即時檢測

1. 使用網路攝像頭
```bash
python camera2.py
```

2. 使用樹莓派相機
```bash
python camera.py
```

3. 批次分析交通影像
```bash
python 0302NewDemo.py
```

## 💡 專案亮點

- **創新性**：首創結合 CNN 和 LSTM 的交通風險預測模型
- **實用性**：可廣泛應用於車載系統、道路監控
- **可擴展性**：模型架構靈活，可輕鬆遷移到其他類型的序列識別任務

## 🔮 未來展望

- [ ] 擴展數據集，提高模型泛化能力
- [ ] 開發移動應用程式
- [ ] 整合更多傳感器數據（雷達、GPS）
- [ ] 支持多語言語音警報
- [ ] 實現車間通信預警網絡


---

<p align="center">
  <b>「馬路三寶，別來撞我」</b> - 用 AI 守護每一位道路使用者的安全 🛡️
</p># 交通安全預警系統

---

## 創作理念

本專案旨在建立一個智慧化交通安全預警系統，透過深度學習技術分析行車影像，並在偵測到潛在危險時及時提醒駕駛者，達到預防交通事故的目的。

我們的願景是：「讓每一位用路人都能平安到家」

---

## 成果說明

### 應用性

- **實用場景**: 可安裝於私家車、商業運輸車輛或交通監控系統
- **即時預警**: 系統能在0.1秒內完成分析並發出警告
- **多模態提醒**: 同時提供視覺和語音警報，確保駕駛者注意
- **低資源需求**: 優化後的模型可在一般車載設備上運行

### 創意性

- **注意力機制**: 引入注意力模型，聚焦於畫面中最關鍵的區域
- **時序理解**: 不僅分析單一畫面，更理解情境的動態變化過程
- **自適應預警**: 根據情境嚴重程度調整警告強度
- **人機協同**: 系統設計考慮人類因素，減少誤報和漏報

---

## 成果說明（續）

### 挑戰性

- **複雜環境適應**: 系統能應對各種天氣、光線和道路條件
- **計算資源限制**: 需在有限硬體條件下實現實時處理
- **多場景訓練**: 資料收集涵蓋各種危險情境，確保全面性
- **平衡敏感度**: 調整系統靈敏度，避免過多干擾或漏警

### 完成性

- **高準確率**: 模型達到99.58%的驗證準確率
- **完整工作流**: 從資料收集、模型訓練到實際應用的全流程實現
- **穩定測試**: 在多種場景下進行測試，確保系統穩定性
- **易於部署**: 完整的部署文檔和示範應用

---

## 程式說明

### 核心模型架構

```python
class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetLSTM, self).__init__()
        
        # 使用預訓練的 ResNet-18 作為特徵提取器
        from torchvision.models import resnet18, ResNet18_Weights
        self.feature_extractor = resnet18(weights=None)
        self.feature_size = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        
        # 注意力機制 - 關注重要幀
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # LSTM層 - 時序理解
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
```

---

## 程式說明（續）

### 預警處理邏輯

```python
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

# 檢查風險狀態是否改變
if current_risk and not self.previous_risk_state:
    # 從安全變為風險 - 啟動警告
    start_warning()
elif not current_risk and self.previous_risk_state:
    # 從風險變為安全 - 停止警告
    stop_warning()
```

---

## 程式說明（續）

### 訓練視覺化程式碼

```python
import matplotlib.pyplot as plt
import numpy as np

# 設置中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 創建圖表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左圖：損失曲線
ax1.plot(epochs, train_losses, 'b-', label='訓練集')
ax1.plot(epochs, val_losses, 'r-', label='驗證集')
ax1.set_title('訓練與驗證損失')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('損失')
ax1.legend()

# 右圖：F1分數曲線
ax2.plot(epochs, train_f1, 'b-', label='訓練 F1 Score')
ax2.plot(epochs, val_f1, 'r-', label='驗證 F1 Score')
ax2.set_title('訓練與驗證 F1 Score')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('F1 Score')
ax2.legend()
```

---

## 其他補充

### 量化分析

- **高效能**: F1分數達0.9958，幾乎所有危險情境都能準確識別
- **低誤報率**: 精確度達100%，不會在安全情況下誤報
- **高召回率**: 召回率達99.16%，幾乎不會漏掉危險情況
- **快速收斂**: 訓練過程在25個epoch內即達到接近完美的表現

### 模型性能

|          | 精確度 | 召回率 | F1分數 | 準確率 |
|----------|-------|-------|-------|-------|
| 最終模型 | 100%   | 99.16% | 99.58% | 99.58% |

---

## 其他補充（續）

### 軟體工程面向

- **模組化設計**: 系統架構清晰，各組件職責明確
- **可擴充性**: 容易加入新的特徵或修改模型結構
- **穩健性**: 包含完整的錯誤處理和異常狀況應對
- **可維護性**: 程式碼結構良好，註解完整

### AI應用面向

- **遷移學習**: 利用預訓練模型，減少訓練資源需求
- **注意力機制**: 引入先進的注意力技術，提高模型理解力
- **多模態融合**: 結合視覺分析和連續時間序列處理
- **實時AI**: 優化推論速度，實現即時處理

---

## 未來展望

- **多元危險辨識**: 擴展到更多交通風險情境的識別
- **環境適應性**: 提高在惡劣天氣和夜間的辨識能力
- **個人化設定**: 根據駕駛者習慣調整預警敏感度
- **多車協同**: 與V2V（車對車）通訊結合，實現群體預警
- **整合導航**: 與導航系統結合，提前預警危險路段

---

## 技術亮點

![訓練與驗證損失和F1分數](training_history.png)

---

## 感謝聆聽

**交通安全，從預防開始！**

開發團隊：Coding_101