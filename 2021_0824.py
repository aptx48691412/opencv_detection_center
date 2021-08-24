import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_center(img):

    y,x=np.where(img==255)
    x_avg,y_avg=np.average(x),np.average(y)
    
    try:
        return [int(x_avg),int(y_avg)]

    except:
        None



def hsv_plot(frame):
    
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame_rgb_hsv=cv2.cvtColor(frame_rgb,cv2.COLOR_BGR2HSV)

    fig=plt.figure()
    axis1=fig.add_subplot(2,2,1)
    axis2=fig.add_subplot(2,2,2)
    axis3=fig.add_subplot(2,2,3)
    axis4=fig.add_subplot(2,2,4)
    axis1.imshow(frame_hsv)
    axis3.imshow(frame_rgb_hsv)

    hsv_list=['h','s','v']
    for ind,i in enumerate(hsv_list):

        calcHist_hsv=cv2.calcHist([frame_hsv],[ind],None,[256],[0,256])
        calcHist_rgb_hsv=cv2.calcHist([frame_rgb_hsv],[ind],None,[256],[0,256])

        axis2.plot(calcHist_hsv,label=i)
        axis4.plot(calcHist_rgb_hsv,label=i)

    axis2.legend()
    axis4.legend()

    plt.tight_layout()
    plt.show()

def hsv_center():
    
    while True:
        ret,frame=cap.read()
        frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        absdiff=cv2.absdiff(frame_hsv,cv2.convertScaleAbs(first_frame))

        ret_,frame_threshold=cv2.threshold(absdiff[:,:,0],0,255,cv2.THRESH_OTSU)
        center=get_center(frame_threshold)
        cv2.circle(frame_threshold,center,35,[0,255,0],45,45)
        cv2.imshow('frame',frame_threshold)

        key=cv2.waitKey(1)
        if key==ord('g'):
            break


def hsv_inRange(first):
    center_list=list()
    while True:
        ret_,frame=cap.read()
        frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #absdiff=cv2.absdiff(frame_hsv,first)
        #lower_filter=(0,0,0)
        #higher_filter=(120,255,255)
        lower_filter = np.array([30, 64, 0])
        higher_filter = np.array([90,255,255])
        inRange_mask=cv2.inRange(frame_hsv,lower_filter,higher_filter)
        bitwise_g=cv2.bitwise_and(frame,frame,mask=inRange_mask)
        gray_g=cv2.cvtColor(bitwise_g,cv2.COLOR_BGR2GRAY)
        ret,threshold_g=cv2.threshold(gray_g,0,255,cv2.THRESH_OTSU)
        morphplogy_threshold=cv2.morphologyEx(threshold_g, cv2.MORPH_OPEN, np.ones((40, 40), np.uint8))
        center_g=get_center(morphplogy_threshold)
        center_list.append(center_g)
        cv2.circle(threshold_g,center_g,5,[0,255,0],5,5)
        cv2.imshow('frame',threshold_g)
        #cv2.imwrite('scscs.png',bitwise_g)

        key=cv2.waitKey(1)

        if key==ord('x'):
            
            break

        elif key==ord('v'):
            center_new_list=[i[0] for i in center_list if i!=None]
            x=np.arange(0,len(center_new_list)*(1/30),1/30)
            plt.plot(x,center_new_list)
            plt.grid()
            #plt.xticks(x*ret)
            plt.xlabel('Time[sec]')
            plt.ylabel('center_placement')
            plt.title('center_placement_activation')
            plt.show()
            #print(center_list[5])












cap=cv2.VideoCapture(0)

index=0
while True:

    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,threshold=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    ret,frame_threshold=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    ret,frame_threshold_hsv=cv2.threshold(frame_hsv[:,:,0],0,255,cv2.THRESH_OTSU)

    if index<20:
        cv2.imshow('frame',frame)
        index+=1

    elif index==20:
        first_frame=frame
        first_frame_gray=cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
        first_frame_hsv=cv2.cvtColor(first_frame,cv2.COLOR_BGR2HSV)
        cv2.imshow('frame',frame)
        index+=1

    else:
        #absdiff=cv2.absdiff(frame,cv2.convertScaleAbs(first_frame))
        #gray_absdiff=cv2.cvtColor(absdiff,cv2.COLOR_BGR2GRAY)
        #ret,absdiff_threshold=cv2.threshold(gray_absdiff,0,255,cv2.THRESH_OTSU)
        #center=get_center(absdiff_threshold)
        center=get_center(threshold)
        cv2.circle(threshold,center,15,[0,255,0],25,25)
        cv2.imshow('frame',threshold)

    key=cv2.waitKey(1)
    if key==27:
        break

    elif key==ord('t'):
        hsv_plot(frame) 

    elif key==ord('s'):
        hsv_center()           
        #break [g]

    elif key==ord('w'):
        hsv_inRange(first_frame)






