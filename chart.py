import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
import platform
import os

# 檢測系統並設置中文字體
system = platform.system()

# 創建字體對象
if system == 'Windows':
    # Windows系統
    font_paths = [
        'C:\\Windows\\Fonts\\msjh.ttc',  # 微軟正黑體
        'C:\\Windows\\Fonts\\mingliu.ttc',  # 細明體
        'C:\\Windows\\Fonts\\kaiu.ttf',  # 標楷體
        'C:\\Windows\\Fonts\\simsun.ttc'  # 新細明體
    ]
elif system == 'Darwin':  # macOS
    font_paths = [
        '/System/Library/Fonts/PingFang.ttc',  # 蘋方
        '/Library/Fonts/Arial Unicode.ttf',  # Arial Unicode MS
        '/Library/Fonts/Heiti.ttc'  # 黑體
    ]
else:  # Linux
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # 文泉驛微米黑
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # Noto Sans CJK
    ]

# 嘗試加載中文字體
chinese_font = None
for font_path in font_paths:
    if os.path.exists(font_path):
        try:
            chinese_font = FontProperties(fname=font_path)
            print(f"成功加載字體: {font_path}")
            break
        except:
            continue

# 如果沒有找到系統字體，使用matplotlib的內建字體
if chinese_font is None:
    print("無法找到系統中文字體，嘗試使用matplotlib內建字體...")
    # 列出所有可用字體
    all_fonts = set([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Microsoft JhengHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    
    for font in chinese_fonts:
        if font in all_fonts:
            chinese_font = FontProperties(fname=matplotlib.font_manager.findfont(font))
            print(f"使用內建字體: {font}")
            break

# 如果還是沒有找到，使用無襯線字體並發出警告
if chinese_font is None:
    print("警告: 無法找到中文字體，圖表中的中文可能無法正確顯示")
    chinese_font = FontProperties(family='sans-serif')

# 設置中文顯示
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 創建數據
epochs = np.arange(0, 30)

# 損失數據
train_losses = [0.68, 0.63, 0.62, 0.66, 0.67, 0.65, 0.67, 0.65, 0.64, 0.6, 0.55, 0.45, 0.38, 0.33, 0.28, 0.25, 0.22, 0.18, 0.16, 0.14, 0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02]
val_losses = [0.52, 0.85, 0.6, 0.63, 0.55, 0.62, 0.56, 0.63, 0.52, 0.4, 0.38, 0.32, 0.22, 0.18, 0.12, 0.19, 0.11, 0.08, 0.09, 0.07, 0.06, 0.08, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02]

# F1分數數據
train_f1 = [0.6, 0.63, 0.69, 0.62, 0.65, 0.69, 0.65, 0.7, 0.72, 0.74, 0.82, 0.85, 0.87, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.96, 0.97, 0.97, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99]
val_f1 = [0.78, 0.73, 0.7, 0.78, 0.69, 0.77, 0.67, 0.7, 0.78, 0.85, 0.87, 0.95, 0.94, 0.95, 0.95, 0.97, 0.96, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]

# 創建圖表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左圖：損失曲線
ax1.plot(epochs, train_losses, 'b-', label='訓練集')
ax1.plot(epochs, val_losses, 'r-', label='驗證集')
ax1.set_title('訓練與驗證損失', fontproperties=chinese_font)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('損失', fontproperties=chinese_font)
ax1.legend(prop=chinese_font)
ax1.set_ylim(0, 0.9)

# 右圖：F1分數曲線
ax2.plot(epochs, train_f1, 'b-', label='訓練 F1 Score')
ax2.plot(epochs, val_f1, 'r-', label='驗證 F1 Score')
ax2.set_title('訓練與驗證 F1 Score', fontproperties=chinese_font)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('F1 Score')
ax2.legend(prop=chinese_font)
ax2.set_ylim(0.6, 1.01)

# 調整佈局
plt.tight_layout()

# 儲存圖表
plt.savefig('training_history2.png', dpi=300, bbox_inches='tight')

# 顯示圖表
plt.show()

# 打印當前可用的字體（調試用）
print("\n可用的字體列表（調試用）:")
for font in sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])[:20]:  # 只顯示前20個
    print(font)