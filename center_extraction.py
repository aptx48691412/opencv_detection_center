import cv2
import numpy as np
import matplotlib.pyplot as plt


cap=cv2.VideoCapture(0)
avg=None
index=0

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #ret,threshold=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

    if index<20:
        index+=1
        

    elif index==20:
        avg=gray.copy().astype('float')
        index+=1
        

    else:
        cv2.accumulateWeighted(gray,avg,0.5)

        absdiff=cv2.absdiff(gray,cv2.convertScaleAbs(avg))
        ret,threshold=cv2.threshold(absdiff,0,255,cv2.THRESH_OTSU)
        cv2.imshow('frame',threshold)

    key=cv2.waitKey(1)

    if key==27:
        break

    if key==ord('g'):
        fig,axis=plt.subplots(2,2)
        #axis[0,0].hist(threshold,bins=45)
        for i in range(2):
            if i==0:
                axis[i,0].imshow(absdiff,cmap='gray')
                axis[i,1].hist(absdiff,bins=45)
            else:
                axis[i,0].imshow(threshold,cmap='gray')
                axis[i,0].set_xlabel('horozontal_pixel')
                axis[i,1].hist(threshold,bins=45)
                axis[i,1].set_xlabel('pixel_value')
            
        #fig=plt.figure(figsize=[480,720])
        #ax0=fig.add_subplot(1,4,1)
        #ax0.imshow(absdiff,cmap='gray')
        #ax1=fig.add_subplot(1,4,2)
        #ax1.hist(absdiff,bins=45) 
        #ax2=fig.add_subplot(1,4,3)
        #ax2.imshow(threshold,cmap='gray')
        #ax3=fig.add_subplot(1,4,4)
        #ax3.hist(threshold,bins=45)   

        plt.tight_layout()
        plt.show()


