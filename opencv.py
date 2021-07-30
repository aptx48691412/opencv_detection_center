import cv2
import numpy as np
cap=cv2.VideoCapture(0)
avg=None
while(True):
    ret, frame = cap.read()    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 前フレームを保存
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    # 現在のフレームと移動平均との間の差を計算する
    # accumulateWeighted関数の第三引数は「どれくらいの早さで以前の画像を忘れるか」。小さければ小さいほど「最新の画像」を重視する。
    # http://opencv.jp/opencv-2svn/cpp/imgproc_motion_analysis_and_object_tracking.html
    # 小さくしないと前のフレームの残像が残る
    # 重みは蓄積し続ける。
    cv2.accumulateWeighted(gray, avg, 0.00001)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]


    # 輪郭を見つける
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #size = 2000
    #list_extracted_contours = extract_contours(contours, size)
    cv2.imshow('frame', thresh)
   
    key=cv2.waitKey(1)
    if key==27:
        break
