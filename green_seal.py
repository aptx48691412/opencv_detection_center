import cv2
import numpy as np
import matplotlib.pyplot as plt


cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    hsv_frame=cv2.cvtColor(rgb_frame,cv2.COLOR_RGB2HSV)



    cv2.imshow('frame',frame)

    key=cv2.waitKey(1)

    if key==27:
        break

    elif key==ord('s'):

        for i in range(3):
            if i==1:

                calcHist_sub=cv2.calcHist([hsv_frame],[i],None,[256],[0,256])    
                plt.plot(calcHist_sub)
                plt.show()
                
            else:
                continue

        


    elif key==ord('y'):

        fig,axis=plt.subplots(2,2)

        axis[0,0].imshow(rgb_frame)
        axis[1,0].imshow(hsv_frame)

        rgb_list=['r','g','b']
        hsv_list=['h','s','v']

        for ind,[i,k] in enumerate(zip(rgb_list,hsv_list)):
            calcHist_rgb=cv2.calcHist([rgb_frame],[ind],None,[256],[0,256])
            calcHist_hsv=cv2.calcHist([hsv_frame],[ind],None,[256],[0,256])
            axis[0,1].plot(calcHist_rgb,c=i,label=i)
            axis[1,1].plot(calcHist_hsv,label=k)

        axis[0,1].legend()
        axis[1,1].legend()
        plt.tight_layout()

        plt.show()






