import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from matplotlib.font_manager import FontProperties

# ======= 中文字體設置方法 1: 直接使用文字檔來繪製中文 =======
def create_text_image():
    """使用純文字來生成中文標題，避免字體問題"""
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
    ax1.plot(epochs, train_losses, 'b-', label='Training')
    ax1.plot(epochs, val_losses, 'r-', label='Validation')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_ylim(0, 0.9)
    
    # 右圖：F1分數曲線
    ax2.plot(epochs, train_f1, 'b-', label='Training F1 Score')
    ax2.plot(epochs, val_f1, 'r-', label='Validation F1 Score')
    ax2.set_title('Training and Validation F1 Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.set_ylim(0.6, 1.01)
    
    # 調整佈局
    plt.tight_layout()
    
    # 儲存基本圖表
    plt.savefig('training_history_base.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 使用PIL添加中文標題
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # 打開剛保存的圖像
        img = Image.open('training_history_base.png')
        draw = ImageDraw.Draw(img)
        
        # 嘗試尋找中文字體
        font_paths = [
            # Windows 字體
            'C:\\Windows\\Fonts\\msjh.ttc',
            'C:\\Windows\\Fonts\\mingliu.ttc',
            'C:\\Windows\\Fonts\\simsun.ttc',
            # macOS 字體
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            # Linux 字體
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, 24)
                    break
                except:
                    continue
        
        # 如果找不到中文字體，使用默認字體
        if font is None:
            font = ImageFont.load_default()
            print("警告: 無法加載中文字體，使用默認字體")
        
        # 添加中文標題
        img_width = img.width
        left_title_pos = (img_width//4, 30)
        right_title_pos = (img_width*3//4, 30)
        
        # 添加白色背景區域來覆蓋原有英文標題
        left_title_bg = (left_title_pos[0]-100, left_title_pos[1]-20, left_title_pos[0]+200, left_title_pos[1]+20)
        right_title_bg = (right_title_pos[0]-100, right_title_pos[1]-20, right_title_pos[0]+200, right_title_pos[1]+20)
        draw.rectangle(left_title_bg, fill="white")
        draw.rectangle(right_title_bg, fill="white")
        
        # 繪製中文標題
        draw.text(left_title_pos, "訓練與驗證損失", fill="black", font=font, anchor="mm")
        draw.text(right_title_pos, "訓練與驗證 F1 Score", fill="black", font=font, anchor="mm")
        
        # 保存最終圖像
        img.save('training_history.png')
        print("成功創建帶中文標題的圖表")
        
    except ImportError:
        print("PIL 庫未安裝，無法添加中文標題")
        print("請安裝 PIL: pip install pillow")

# ======= 中文字體設置方法 2: 使用matplotlib內建方法 =======
def matplotlib_with_chinese():
    """使用matplotlib內建方法顯示中文"""
    # 嘗試設置中文字體
    try:
        # 解決中文顯示問題
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
        plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
        
        # 檢查系統中是否有中文字體
        # 嘗試查找常見中文字體
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
        all_fonts = set([f.name for f in matplotlib.font_manager.fontManager.ttflist])
        
        found_font = False
        for font in chinese_fonts:
            if font in all_fonts:
                plt.rcParams['font.sans-serif'].insert(0, font)
                found_font = True
                print(f"使用中文字體: {font}")
                break
        
        if not found_font:
            # 嘗試使用系統字體路徑
            import platform
            system = platform.system()
            
            if system == 'Windows':
                font_path = 'C:\\Windows\\Fonts\\simsun.ttc'  # 新細明體
            elif system == 'Darwin':  # macOS
                font_path = '/System/Library/Fonts/PingFang.ttc'  # 蘋方
            else:  # Linux
                font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # 文泉驛微米黑
            
            if os.path.exists(font_path):
                from matplotlib.font_manager import FontProperties
                chinese_font = FontProperties(fname=font_path)
                print(f"使用系統字體: {font_path}")
            else:
                print("警告: 無法找到系統中文字體，可能無法正確顯示中文")
    except Exception as e:
        print(f"設置中文字體時出錯: {str(e)}")
    
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
    ax1.set_title('訓練與驗證損失')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('損失')
    ax1.legend()
    ax1.set_ylim(0, 0.9)
    
    # 右圖：F1分數曲線
    ax2.plot(epochs, train_f1, 'b-', label='訓練 F1 Score')
    ax2.plot(epochs, val_f1, 'r-', label='驗證 F1 Score')
    ax2.set_title('訓練與驗證 F1 Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.set_ylim(0.6, 1.01)
    
    # 調整佈局
    plt.tight_layout()
    
    # 儲存圖表
    plt.savefig('training_history_matplotlib.png', dpi=300, bbox_inches='tight')
    plt.show()

# 運行兩種方法，確保至少一種能正確顯示中文
print("嘗試使用兩種方法生成帶中文標題的圖表...")
try:
    matplotlib_with_chinese()
    print("使用matplotlib方法生成圖表成功!")
except Exception as e:
    print(f"matplotlib方法失敗: {str(e)}")
    try:
        create_text_image()
        print("使用PIL方法生成圖表成功!")
    except Exception as e:
        print(f"PIL方法失敗: {str(e)}")
        print("無法生成帶中文標題的圖表，請檢查系統字體設置。")