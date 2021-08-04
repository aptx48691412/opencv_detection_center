import numpy as np
import cv2
from matplotlib import pyplot as plt

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

    if key==ord('h'):
        # global thresholding
        # ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

        # Otsu's thresholding
        ret2,th2 = cv2.threshold(diff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(diff,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # plot all the images and their histograms
        images = [gray, 0, th2,
                  blur, 0, th3,
                  th2, 0, th2,
                  th3, 0, th3]
        titles = ['Original Noisy Image','Histogram',"Otsu's Thresholding",
                  'Gaussian filtered Image','Histogram',"Otsu's Thresholding",
                  "Otsu's Thresholding",'Histogram',"Otsu's Thresholding",
                  "Otsu's Thresholding",'Histogram',"Otsu's Thresholding"]
        
        fig,axis=plt.subplots(4,3)
  

        for i in range(4):
            axis[i,0].imshow(images[i*3],'gray')
            #axis[i,0].title(titles[i*3])
            #axis[i,0].xticks([])
            #axis[i,0].yticks([])
            #plt.subplot(4,3,i*3+2)
            axis[i,1].hist(images[i*3].ravel(),256)
            #axis[i,1].title(titles[i*3+1])
            #axis[i,1].xticks([])
            #axis[i,1].yticks([])
            #plt.subplot(4,3,i*3+3)
            axis[i,2].imshow(images[i*3+2],'gray')
            #axis[i,2].title(titles[i*3+2])
            #axis[i,2].xticks([])
            #axis[i,2].yticks([])
        plt.figure()
        plt.show()
        







   


    


     
