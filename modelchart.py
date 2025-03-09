import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np

# 設定中文字體支援
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 創建畫布
plt.figure(figsize=(18, 9))
ax = plt.gca()

# 設置顏色
resnet_color = '#9FD5F0'  # 淺藍色
lstm_color = '#A0EDA3'    # 淺綠色
fc_color = '#FFABA8'      # 淺紅色
attention_color = '#FFF3B8'  # 淺黃色
arrow_color = '#303030'   # 深灰色
text_color = '#000000'    # 黑色

# 繪製各區塊，確保足夠大小
# 繪製ResNet特徵提取器區塊
resnet_block = patches.FancyBboxPatch(
    (1.0, 1.5), 4.0, 5.0, 
    boxstyle=patches.BoxStyle("Round", pad=0.6),
    facecolor=resnet_color, alpha=0.9, edgecolor='black', linewidth=2
)
ax.add_patch(resnet_block)

# 繪製注意力機制區塊
attention_block = patches.FancyBboxPatch(
    (7.0, 1.5), 4.0, 5.0, 
    boxstyle=patches.BoxStyle("Round", pad=0.6),
    facecolor=attention_color, alpha=0.9, edgecolor='black', linewidth=2
)
ax.add_patch(attention_block)

# 繪製LSTM區塊
lstm_block = patches.FancyBboxPatch(
    (13.0, 1.5), 4.0, 5.0, 
    boxstyle=patches.BoxStyle("Round", pad=0.6),
    facecolor=lstm_color, alpha=0.9, edgecolor='black', linewidth=2
)
ax.add_patch(lstm_block)

# 繪製全連接層區塊
fc_block = patches.FancyBboxPatch(
    (19.0, 1.5), 4.0, 5.0, 
    boxstyle=patches.BoxStyle("Round", pad=0.6),
    facecolor=fc_color, alpha=0.9, edgecolor='black', linewidth=2
)
ax.add_patch(fc_block)

# 添加箭頭 - 清晰連接各區塊
# ResNet到Attention
plt.arrow(5.2, 4.0, 1.6, 0, head_width=0.2, head_length=0.3, 
          fc=arrow_color, ec=arrow_color, linewidth=2.5)

# Attention到LSTM
plt.arrow(11.2, 4.0, 1.6, 0, head_width=0.2, head_length=0.3, 
          fc=arrow_color, ec=arrow_color, linewidth=2.5)

# LSTM到FC
plt.arrow(17.2, 4.0, 1.6, 0, head_width=0.2, head_length=0.3, 
          fc=arrow_color, ec=arrow_color, linewidth=2.5)

# 添加標籤 - 只保留重點，確保在框內
# ResNet標籤
plt.text(3.0, 5.8, "ResNet特徵提取", ha='center', fontsize=28, fontweight='bold', color=text_color)
plt.text(3.0, 4.8, "• ResNet-18", ha='center', fontsize=22, color=text_color)
plt.text(3.0, 4.0, "• Identity()", ha='center', fontsize=22, color=text_color)
plt.text(3.0, 3.2, "• 空間特徵提取", ha='center', fontsize=22, color=text_color)
plt.text(3.0, 2.2, "輸出維度: 512", ha='center', fontsize=22, color=text_color)

# 注意力機制標籤
plt.text(9.0, 5.8, "注意力機制", ha='center', fontsize=28, fontweight='bold', color=text_color)
plt.text(9.0, 4.8, "• Linear + Tanh", ha='center', fontsize=22, color=text_color)
plt.text(9.0, 4.0, "• Linear + Softmax", ha='center', fontsize=22, color=text_color)
plt.text(9.0, 3.2, "• 識別關鍵幀", ha='center', fontsize=22, color=text_color)
plt.text(9.0, 2.2, "上下文向量生成", ha='center', fontsize=22, color=text_color)

# LSTM標籤
plt.text(15.0, 5.8, "雙向LSTM", ha='center', fontsize=28, fontweight='bold', color=text_color)
plt.text(15.0, 4.8, "• 輸入維度: 512", ha='center', fontsize=22, color=text_color)
plt.text(15.0, 4.0, "• 隱藏層: 256", ha='center', fontsize=22, color=text_color)
plt.text(15.0, 3.2, "• 雙向、兩層", ha='center', fontsize=22, color=text_color)
plt.text(15.0, 2.2, "Dropout: 0.5", ha='center', fontsize=22, color=text_color)

# FC標籤
plt.text(21.0, 5.8, "全連接分類器", ha='center', fontsize=28, fontweight='bold', color=text_color)
plt.text(21.0, 4.8, "• 512 → 128", ha='center', fontsize=22, color=text_color)
plt.text(21.0, 4.0, "• 128 → 64", ha='center', fontsize=22, color=text_color)
plt.text(21.0, 3.2, "• 64 → 1", ha='center', fontsize=22, color=text_color)
plt.text(21.0, 2.2, "風險機率輸出", ha='center', fontsize=22, color=text_color)

# 設置軸的範圍和隱藏軸
plt.axis('equal')
plt.axis('off')
plt.xlim(0, 24)
plt.ylim(0, 7.5)

# 添加標題
plt.title("ResNet+LSTM 交通風險預測模型架構", fontsize=22, fontweight='bold')


# 保存圖像
plt.tight_layout()
plt.savefig("resnet_lstm_architecture.svg", format='svg', bbox_inches='tight', facecolor='white')

# 顯示圖像
plt.show()