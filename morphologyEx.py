#from sys import platform
from typing import get_args
import cv2
import numpy as np
import matplotlib.pyplot as plt


cap=cv2.VideoCapture(0)
index=0


def get_center(img):

    y,x = np.where(img==255)
    x_avg,y_avg=np.average(x),np.average(y)
    return [int(x_avg),int(y_avg)]



while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((50, 50), np.uint8))
    if index<20:
        cv2.imshow('frame',gray)
        index+=1
    
    elif index==20:

        first_frame=gray.copy().astype('float')
        cv2.imshow('frame',gray)
        index+=1

    else:
        #cv2.accumulateWeighted(gray,first_frame,0.3)
        absdiff=cv2.absdiff(gray,cv2.convertScaleAbs(first_frame))
        ret,threshold=cv2.threshold(absdiff,0,255,cv2.THRESH_OTSU)
        #cv2.threshold(gray,first_frame)



        center=get_center(threshold)

        cv2.circle(threshold,center,10,[0,255,0],35,35)
        cv2.imshow('frame',threshold)
        

    key=cv2.waitKey(1)
    if key==27:
        break

    if key==ord('r'):

        fig=plt.figure()
        axis1,axis2=fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
        axis1.imshow(threshold)
        axis2.hist(threshold,bins=60)





