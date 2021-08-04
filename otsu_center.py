import numpy as np
import cv2

cap=cv2.VideoCapture(0)

avg=0

while True:

    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #ret,threshold=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    #ret,threshold=cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    
    if avg==20:
        first_frame=gray.copy().astype("float")
        cv2.imshow('Frame',gray)
        avg+=1

    elif avg>20:
        cv2.accumulateWeighted(gray,first_frame,0.3)
        diff=cv2.absdiff(gray,cv2.convertScaleAbs(first_frame))
        ret,threshold=cv2.threshold(diff,0,255,cv2.THRESH_OTSU)
        cv2.imshow('Frame',threshold)
    
    else:
        avg+=1
        cv2.imshow('Frame',gray)
        


    key=cv2.waitKey(1)
    if key==27:
        break

    if key==ord('u'):
        print('Amazing')






   


    


     
