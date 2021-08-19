import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.core.defchararray import index

cap=cv2.VideoCapture(0)
#print(cap.get(cv2.CAP_PROP_FPS))
center_list=list()
fps_list=list()
index=0

def getCenter(binimg):
    ys, xs= np.where(binimg == 255)
    x = np.average(xs)
    y = np.average(ys)
    return [int(x),int(y)]

while True:

    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    exact_gray=cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    
    #ret,threshold=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    #ret,threshold=cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    #ret,frame=cv2.threshold(gray,0,255,cv2.THRESH_BINARY)



    if index==20:
        first_frame=exact_gray.copy().astype("float")
        #cv2.imshow('Frame',gray)
        index+=1

    elif index>20:
        #cv2.accumulateWeighted(exact_gray,first_frame,0.2)
        absdiff=cv2.absdiff(exact_gray,cv2.convertScaleAbs(first_frame))
        ret,threshold=cv2.threshold(absdiff,0,255,cv2.THRESH_OTSU)
        #ret,frame=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
        #gray=cv2.cvtColor(frame,cv2.COLOR_THREDH_OTSU)

        #cv2.imshow('frame',frame)
        

        center=getCenter(threshold)
        center_list.append(center)
        fps_list.append(cap.get(cv2.CAP_PROP_FPS))
        #cv2.imshow('Frame',threshold)
        cv2.circle(frame,center,20,[34,255,34],40,20)
        cv2.circle(exact_gray,center,20,[34,255,34],40,20)
        
        cv2.circle(threshold,center,5,[34,255,34],40,20)
        
        cv2.imshow('Frame',threshold)
        #print(cap.get(cv2.CAP_PROP_FPS))
        
        #cv2.imshow('threshold',threshold)




        

    
    else:
        index+=1
        cv2.imshow('Frame',exact_gray)
        


    key=cv2.waitKey(1)
    if key==27:
        break

    if key==ord('u'):
        print('Amazing')
        print(absdiff.ravel())
        print('-----------')
        print(type(absdiff.ravel()))
        print('-----------')
        print(absdiff.ravel().shape)
        #plt.hist(absdiff.ravel(),35)
        #plt.show()
        fig,axis=plt.subplots(2,3)
        #axis[0,0].hist(np.array([1,2,1,1,1,2,2,2,2,2]),bins=67)
        axis[0,0].imshow(frame)
        #plt.hist(np.array([1,2,1,1,1,2,2,2,2,2]),bins=67)
        plt.show()
        

    if key==ord('w'):
        #cv2.imshow('threshold',threshold)
        #for i in center_list:

        #y,x=getCenter(threshold==255)
        
        fig=plt.figure()
        axis=fig.add_subplot(1,2,1)
        axis.imshow('aaaaa',threshold)
        plt.title(threshold.shape)
        plt.show()




    if key==ord('c'):
        #fig,axis=plt.subplots(1,3)
        #axis[0,0].hist(absdiff.ravel(),bins=56)
        #axis[0,0].hist(np.array([0,1,2,1,2,1,0,2,2,2]),bins=50)
        #axis[0,1].imshow('absdiff',absdiff)
        #axis[0,2].imshow('threshold',threshold)
        #plt.hist(absdiff.ravel(),bins=255)
        #plt.hist(threshold.ravel(),bins=255)
        plt.show()

        #center=getCenter(threshold)
        #print(center)
        fig,axis=plt.subplots(4,2)
        #print(absdiff)

        img_list=[cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),exact_gray,absdiff,threshold]
        img_name_list=['frame','gray','absdiff','threshold']

        for i in range(4):
            if i==0:
                axis[i,0].imshow(img_list[i])
            else:
                axis[i,0].imshow(img_list[i],cmap='gray')
    
            axis[i,0].set_title(img_name_list[i])
            axis[i,1].hist(img_list[i].ravel(),100)
            axis[i,1].set_xlabel('pixel_value')
            axis[i,1].set_ylabel('number_of_pixel')
            

        plt.tight_layout()    
        fig.savefig('center_placemnet.png')


        #axis[0,2].imshow(threshold,'threshold')
        #axis[1,0].hist(diff.ravel(),256)
        #axis[1,1].imshow(diff,'diff')
        #axis[1,2].imshow(threshold,'threshold')

        #axis[1,0].hist(threshold.ravel(),34)
        #axis[1,1].plot()
        #axis[1,2].plot()
        #axis[2,0].hist()


        plt.show()

    if key==ord('j'):
        print(center_list)
        print(np.average([i[0] for i in center_list]))


    if key==ord('e'):
        fig=plt.figure()
        #ax1=fig.add_subplot(1,2,1)
        #ax2=fig.add_subplot(1,2,2)
        #ax1.imshow(threshold,cmap='gray')
        #ax2.hist(threshold,bins=128)
        plt.imshow(threshold,cmap='gray')
        plt.xlabel('the placement of center is  {} {}'.format(center,ret))
        #plt.title(ret)

        #fig.savefig('threshold.png')
        plt.show()
        list__=np.where(threshold==255)
        #print(list__)
        #print(center[1])
        v_list=list()
        for ii,i in enumerate(center_list):
            
            if ii==0:
                v_list.append(0)
                kkk=i
            else:
                v_list.append(abs(i[0]-kkk[0]))
                kkk=i

        print(v_list)
        print('-----------------')        
        print(np.average(v_list)/np.average(fps_list))

        #fig=plt.figure()
        #plt.hist(threshold.ravel(),bins=128)
        #fig.savefig('threshold_hist.png')
        #plt.show()

        #fig.savefig('threshold_new.png')


    if key==ord('n'):
        print(np.average(fps_list))
        print('ytdkflcv')

    if key==ord('m'):
        #img_=cv2.imread('threshold.png')
        #list_=np.where(img_==255)
        #print(len(list_))
        cv2.imshow('img_',threshold)
        plt.show()
        print(threshold.shape)
        print(np.average(fps_list))











    if key==ord('h'):
        # global thresholding
        # ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

        # Otsu's thresholding
        print(getCenter(threshold))

        ret2,th2 = cv2.threshold(absdiff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(absdiff,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # plot all the images and their histograms
        images = [gray, 0, th2,
                  blur, 0, th3,
                  th2, 0, th2,
                  th3, 0, th3,
                  absdiff,0,threshold]
        titles = ['Original Noisy Image','Histogram',"Otsu's Thresholding",
                  'Gaussian filtered Image','Histogram',"Otsu's Thresholding",
                  "Otsu's Thresholding",'Histogram',"Otsu's Thresholding",
                  "Otsu's Thresholding",'Histogram',"Otsu's Thresholding",
                  'Original Noisy Image','Histogram',"Otsu's Thresholding"]
        
        fig,axis=plt.subplots(5,3)
  

        for i in range(5):
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
        plt.tight_layout()
        #fig=plt.figure()
        #fig.savefig('arm_hist.png')
        #cv2.circle(absdiff,getCenter(threshold))
        print(images[4*3])
        print(images[4*3].shape)
        print(absdiff.shape)

        plt.show()
        
        







   


    


     
