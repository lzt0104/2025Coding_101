import cv2
print("正在嘗試開啟相機...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法開啟相機索引0")
else:
    print("成功開啟相機")
    ret, frame = cap.read()
    if ret:
        print("成功讀取一幀影像")
        cv2.imwrite("test_frame.jpg", frame)
        print("已保存測試圖像到test_frame.jpg")
    else:
        print("讀取影像失敗")
cap.release()