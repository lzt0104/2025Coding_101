import RPi.GPIO as GPIO
import time

# 设置GPIO模式为BCM
GPIO.setmode(GPIO.BCM)

# 定义SW-420的DO引脚
VIBRATION_SENSOR_PIN = 17

# 设置GPIO引脚为输入模式，并启用内部下拉电阻
GPIO.setup(VIBRATION_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


try:
    print("开始检测振动...")
    while True:
        # 读取传感器的状态
        vibration_state = GPIO.input(VIBRATION_SENSOR_PIN)
        print(vibration_state)

        # if vibration_state == GPIO.HIGH:
        #     print("检测到振动！")
        # else:
        #     print("无振动")

        # # 每隔0.5秒检测一次
        time.sleep(0.5)

except KeyboardInterrupt:
    print("程序终止")

finally:
    # 清理GPIO设置
    GPIO.cleanup()
    
    

    
    
