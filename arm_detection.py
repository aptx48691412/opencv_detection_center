import cv2
cap=cv2.VideoCapture(0)
avg=None
#img=cv2.imread(r'/Users/yamauchisachiyo/Desktop/Lenna.png',cv2.IMREAD_GRAYSCALE)

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if avg is None:
        avg = gray.copy().astype("float")
        continue

    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, avg, 0.6)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    ret_th,threshold=cv2.threshold(frameDelta,3,255,cv2.THRESH_BINARY)
    contours,hierarchy=cv2.findContours(threshold,1,2)

    #for i in contours:
    #    print(i[1])
    #print('-----------')
    if len(contours)>5:
        for index,i in enumerate(contours):
            if index==0:
                contours_area_max=cv2.contourArea(i)
                contours_max=i

            else:
                if cv2.contourArea(i)>contours_area_max:
                    contours_max=i
                else:
                    continue

    else:
        continue


    contours_moments=cv2.moments(contours_max)
    x,y= int(contours_moments["m10"]/contours_moments["m00"]) , int(contours_moments["m01"]/contours_moments["m00"])
    
    cv2.circle(frame, (x,y), 20, 255, 5, 24)
    cv2.drawContours(frame,contours_max,-1,(0, 255, 0), 3)
    cv2.imshow('Contours & Moments',frame)
    key=cv2.waitKey(1)
    if key==27:
        break

    elif key==ord('y'):
        print(x,y)
        print(contours_max)
