import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import cv2
import random
from sklearn.model_selection import train_test_split

# 設置隨機種子，確保可重現性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# 設置記憶體優化環境變數
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 改進版 CNN-LSTM 模型
class ImprovedCNN_LSTM(nn.Module):
    def __init__(self, num_classes=1, seq_length=10):
        super(ImprovedCNN_LSTM, self).__init__()
        
        # 使用預訓練的 ResNet-18 作為特徵提取器
        # 移除最後的全連接層
        from torchvision.models import resnet18, ResNet18_Weights
        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_size = self.feature_extractor.fc.in_features  # 通常是 512
        self.feature_extractor.fc = nn.Identity()  # 移除最後的全連接層
        
        # 凍結前幾層以避免過擬合
        layers_to_freeze = list(self.feature_extractor.children())[:6]  # 凍結前6層
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        
        # LSTM 部分
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        # 注意力機制
        self.attention = nn.Sequential(
            nn.Linear(512, 128),  # 512 = 256*2 (雙向LSTM)
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # 分類器
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
        
        # 時間序列長度
        self.seq_length = seq_length
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # 展平時間維度，作為批次處理
        x = x.view(batch_size * seq_len, C, H, W)
        
        # 提取特徵
        x = self.feature_extractor(x)  # 輸出: [batch_size*seq_len, feature_size]
        
        # 重塑為序列格式
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, feature_size]
        
        # 通過 LSTM 處理序列
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # 應用注意力機制
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size*2]
        
        # 最終分類
        output = self.classifier(context_vector)
        
        return output

# 真實交通數據集
class TrafficDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None, max_sequences=None, mode='train'):
        """
        真實交通數據集
        
        Args:
            root_dir (str): 資料夾路徑，包含多個子資料夾，每個子資料夾是一個序列
            sequence_length (int): 每個樣本的時間步數
            transform: 圖像轉換
            max_sequences (int, optional): 限制使用的序列數量
            mode (str): 'train' 或 'test'，影響資料增強和處理
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.mode = mode
        
        # 獲取所有序列資料夾
        self.sequence_folders = sorted([
            f for f in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, f)) and f.startswith('freeway_')
        ])
        
        if max_sequences and max_sequences < len(self.sequence_folders):
            # 隨機選擇子集
            random.shuffle(self.sequence_folders)
            self.sequence_folders = self.sequence_folders[:max_sequences]
            
        print(f"找到 {len(self.sequence_folders)} 個序列資料夾")
        
        # 為每個序列加載標籤（這裡需要根據實際情況調整）
        self.sequence_labels = {}
        self.sequences = []
        
        for folder in self.sequence_folders:
            folder_path = os.path.join(root_dir, folder)
            
            # 獲取所有圖像並按數字排序
            images = sorted([
                f for f in os.listdir(folder_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ], key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
            
            # 如果圖像太少，跳過
            if len(images) < sequence_length:
                continue
                
            # 為每個可能的子序列創建一個樣本
            for i in range(0, len(images) - sequence_length + 1, sequence_length // 2):  # 使用重疊窗口
                sub_sequence = images[i:i+sequence_length]
                if len(sub_sequence) == sequence_length:
                    # 這裡需要為每個子序列分配標籤
                    # 例如，可以基於圖像特徵或文件名來確定標籤
                    # 對於演示，我們使用簡單的啟發式方法：
                    # 如果文件夾名稱的數字是奇數，標記為正樣本，否則為負樣本
                    folder_id = int(folder.split('_')[1])
                    label = 1 if folder_id % 2 == 1 else 0  # 奇數為正樣本
                    
                    self.sequences.append({
                        'folder': folder_path,
                        'images': sub_sequence,
                        'label': label
                    })
        
        # 確保類別平衡
        positive_samples = [s for s in self.sequences if s['label'] == 1]
        negative_samples = [s for s in self.sequences if s['label'] == 0]
        
        # 通過下采樣較多的類別來平衡數據集
        min_samples = min(len(positive_samples), len(negative_samples))
        
        if len(positive_samples) > min_samples:
            positive_samples = random.sample(positive_samples, min_samples)
        if len(negative_samples) > min_samples:
            negative_samples = random.sample(negative_samples, min_samples)
            
        self.sequences = positive_samples + negative_samples
        random.shuffle(self.sequences)
        
        print(f"創建了 {len(self.sequences)} 個樣本 (正樣本: {len(positive_samples)}, 負樣本: {len(negative_samples)})")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        folder_path = sequence_info['folder']
        image_names = sequence_info['images']
        label = sequence_info['label']
        
        # 加載並轉換圖像序列
        images = []
        for img_name in image_names:
            img_path = os.path.join(folder_path, img_name)
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            images.append(image)
        
        # 堆疊為單一張量
        image_sequence = torch.stack(images)  # [sequence_length, C, H, W]
        
        return image_sequence, torch.FloatTensor([float(label)])

# 評估函數
def evaluate_model(model, data_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="評估中"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 計算評估指標
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return metrics

# 訓練函數
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, 
               num_epochs=30, patience=7, checkpoint_path='best_traffic_model.pth'):
    # 初始化變數
    best_val_f1 = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 修正 autocast 用法，加入必要的 device_type 參數
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type, enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()
            
            train_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())
        
        # 計算訓練指標
        train_loss = train_loss / len(train_loader)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            all_train_labels, all_train_preds, average='binary', zero_division=0
        )
        train_accuracy = accuracy_score(all_train_labels, all_train_preds) * 100
        
        # 驗證階段
        val_metrics = evaluate_model(model, val_loader, criterion)
        
        # 記錄指標
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        train_f1s.append(train_f1)
        val_f1s.append(val_metrics['f1'])
        
        # 打印訓練資訊
        print(f"Epoch {epoch+1}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}%, F1: {train_f1:.4f}")
        print(f"Valid - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, F1: {val_metrics['f1']:.4f}")
        print(f"Valid - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")
        
        # 檢查是否為最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_f1,
                'accuracy': val_metrics['accuracy'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall']
            }, checkpoint_path)
            
            print(f"儲存最佳模型 (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            
        # 提前停止
        if patience_counter >= patience:
            print(f"驗證 F1 已 {patience} 個 epoch 未改善，提前停止訓練")
            break
            
        print("-" * 60)
    
    # 繪製訓練歷史
    plot_training_history(train_losses, val_losses, train_f1s, val_f1s, 
                        save_path='training_history.png', metric_name='F1 Score')
    
    return train_losses, val_losses, train_f1s, val_f1s

# 繪製訓練歷史
def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, 
                         save_path='training_history.png', metric_name='Accuracy'):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='訓練損失')
    plt.plot(epochs, val_losses, 'r-', label='驗證損失')
    plt.title('訓練與驗證損失')
    plt.xlabel('Epochs')
    plt.ylabel('損失')
    plt.legend()
    
    # 指標曲線
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metrics, 'b-', label=f'訓練 {metric_name}')
    plt.plot(epochs, val_metrics, 'r-', label=f'驗證 {metric_name}')
    plt.title(f'訓練與驗證 {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"訓練歷史已儲存至 {save_path}")

# 主程式
def main():
    # 配置參數
    data_root = "train/train"  # 根據實際路徑調整
    sequence_length = 10
    batch_size = 100
    num_epochs = 30
    learning_rate = 0.0005
    num_classes = 1
    
    # 圖像轉換
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 載入資料集
    print("載入資料集...")
    
    # 檢查數據集路徑是否存在
    if not os.path.exists(data_root):
        potential_paths = [
            "train/train",
            "train",
            "data/train",
            "dataset/train",
            "../train/train",
            "../train"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                data_root = path
                print(f"找到數據集路徑: {data_root}")
                break
        else:
            print("無法找到數據集路徑，請手動指定")
            return
    
    # 創建完整數據集
    full_dataset = TrafficDataset(root_dir=data_root, 
                                sequence_length=sequence_length, 
                                transform=None,  # 先不應用轉換
                                max_sequences=None)  # 使用所有數據
    
    # 分割數據集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=val_size/len(full_dataset),
        random_state=42,
        stratify=[s['label'] for s in full_dataset.sequences]  # 確保分層抽樣
    )
    
    # 創建訓練和驗證數據集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # 添加數據轉換
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # 數據加載器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"訓練數據集大小: {len(train_dataset)}")
    print(f"驗證數據集大小: {len(val_dataset)}")
    
    # 初始化模型
    model = ImprovedCNN_LSTM(num_classes=num_classes, seq_length=sequence_length).to(device)
    
    # 統計模型參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數數量: {total_params:,}")
    print(f"可訓練參數數量: {trainable_params:,}")
    
    # 初始化損失函數和優化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 學習率調度器
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # 開始訓練
    print("開始訓練...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=num_epochs,
        patience=7,
        checkpoint_path='best_traffic_model.pth'
    )
    
    # 載入最佳模型進行最終評估
    print("載入最佳模型進行最終評估...")
    checkpoint = torch.load('best_traffic_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最終評估
    val_metrics = evaluate_model(model, val_loader, criterion)
    
    # 輸出最終結果
    print("\n最終評估結果:")
    print(f"準確率: {val_metrics['accuracy']:.2f}%")
    print(f"精確度: {val_metrics['precision']:.4f}")
    print(f"召回率: {val_metrics['recall']:.4f}")
    print(f"F1 分數: {val_metrics['f1']:.4f}")
    print(f"混淆矩陣:\n{val_metrics['confusion_matrix']}")
    
    # 保存完整模型，使用 state_dict 保存以便於移植性
    torch.save(model.state_dict(), 'traffic_model_state_dict.pth')
    
    # 保存 scriptable 模型版本 (TorchScript)，便於部署
    scripted_model = torch.jit.script(model.cpu())
    scripted_model.save('traffic_model_scripted.pt')
    
    print("訓練完成！所有模型檔案已保存。")
    
if __name__ == "__main__":
    main()