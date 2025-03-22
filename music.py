import pyttsx3
import os

# 創建輸出目錄
output_dir = "voice_alerts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 要生成的語音提示
alert_text = "Plaese Reduce the speed"

# 初始化 pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 200)  # 調整語速
engine.setProperty('volume', 1.0)  # 調整音量

# 保存語音提示
for text, name in [(alert_text, "slow_down")]:
    # 保存為WAV文件
    wav_path = os.path.join(output_dir, f"{name}.wav")
    engine.save_to_file(text, wav_path)
    engine.runAndWait()
    print(f"已保存語音到 {wav_path}")

print("語音文件創建完成！")