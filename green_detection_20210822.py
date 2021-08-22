import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
import pandas as pd
import requests


def get_center(img):
    y,x=np.where(img==255)
    x_avg,y_avg=np.average(x),np.average(y)

    return [int(x_avg),int(y_avg)]

def detect_green_color(frame_def):
        # HSV色空間に変換
    hsv = cv2.cvtColor(frame_def, cv2.COLOR_BGR2HSV)

    # 緑色のHSVの値域1
    hsv_min = np.array([30, 64, 0])
    hsv_max = np.array([90,255,255])

    # 緑色領域のマスク（255：赤色、0：赤色以外）    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    
    # マスキング処理
    masked_img = cv2.bitwise_and(frame_def, frame_def, mask=mask)

    return mask, masked_img




def extraction_img(url):
    resp= requests.get(url, stream=True).raw
    image= np.asarray(bytearray(resp.read()), dtype="uint8")
    image= cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def hsv_convert(frame_default):
    
    frame_default_rgb=cv2.cvtColor(frame_default,cv2.COLOR_BGR2RGB)
    global frame_default_hsv
    frame_default_hsv=cv2.cvtColor(frame_default,cv2.COLOR_BGR2HSV)
    global frame_default_rgb_hsv
    frame_default_rgb_hsv=cv2.cvtColor(frame_default_rgb,cv2.COLOR_BGR2HSV)

    ret_hsv,threshold_hsv=cv2.threshold(frame_default_hsv[:,:,0],0,256,cv2.THRESH_OTSU)
    ret_rgb_hsv,threshold_rgb_hsv=cv2.threshold(frame_default_rgb_hsv[:,:,0],0,256,cv2.THRESH_OTSU)
    global morphology_hsv
    morphology_hsv=cv2.morphologyEx(threshold_hsv, cv2.MORPH_OPEN, np.ones((20, 20), np.uint8))
    global morphology_rgb_hsv
    morphology_rgb_hsv=cv2.morphologyEx(threshold_rgb_hsv, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
    
def hsv_plot():
    
    fig,axis=plt.subplots(3,2)

    axis[0,0].imshow(frame_default_hsv[:,:,0])
    axis[1,0].imshow(frame_default_rgb_hsv[:,:,0])

    hsv_list=['h','s','v']

    for ind,i in enumerate(hsv_list):
        if ind==0:


            calcHist_hsv=cv2.calcHist([frame_default_hsv],[ind],None,[256],[0,256])
            calcHist_rgb_hsv=cv2.calcHist([frame_default_rgb_hsv],[ind],None,[256],[0,256])
            axis[0,1].plot(calcHist_hsv,label=i)
            axis[1,1].plot(calcHist_rgb_hsv,label=i)

        else:
            continue

    axis[0,1].legend()
    axis[1,1].legend()

    axis[2,0].imshow(morphology_hsv,cmap='gray')
    axis[2,1].imshow(cv2.bitwise_not(morphology_rgb_hsv),cmap='gray')
    
    plt.tight_layout()

    plt.show()

def center_placement(center,ret):

    x_center=np.average([i_[0] for i_ in center])
    y_center=np.average([k_[1] for k_ in center])
    ret_average=np.average(ret)

    print('X_Average_v = {} // Y_Average_v = {}'.format(x_center/ret_average,y_center/ret_average))



cap=cv2.VideoCapture(0)
while True:

    ret,frame=cap.read()
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret_,threshold=cv2.threshold(gray,0,256,cv2.THRESH_OTSU)

    frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #frame_hsv=cv2.imread('/Users/yamauchisachiyo/Downloads/Lenna.png')
    #frame_hsv=cv2.cvtColor(frame_hsv,cv2.COLOR_BGR2HSV)
    
    
    
    frame_rgb_hsv=cv2.cvtColor(frame_rgb,cv2.COLOR_BGR2HSV)

    cv2.imshow('frame',frame)

    key=cv2.waitKey(1)
    if key==27:
        break

    elif key==ord('e'):
        #frame=extraction_img('https://auctions.c.yimg.jp/images.auctions.yahoo.co.jp/image/dr000/auc0508/users/c1723af7f6ed08ebaba9b214afefb52400b60341/i-img900x1200-1565502088o5ro2g914284.jpg')
        
        center_list=list()
        ret_e_list=list()
        
        while True:
            ret_e,frame_e=cap.read()
            ret_e_list.append(ret_e)
        
            hsv_convert(frame_e)
            cv2.imshow('frame',morphology_hsv)

            center=get_center(morphology_hsv)
            center_list.append(center)
            #print(center)



            key__=cv2.waitKey(1)

            if key__==ord('n'):
                break

            elif key__==ord('t'):
                center_placement(center_list,ret_e_list)

                




        


    elif key==ord('g'):
        #red_mask, red_masked_img = detect_red_color(img)
        img=extraction_img('https://auctions.c.yimg.jp/images.auctions.yahoo.co.jp/image/dr000/auc0508/users/c1723af7f6ed08ebaba9b214afefb52400b60341/i-img900x1200-1565502088o5ro2g914284.jpg')
        green_mask,green_masked_img = detect_green_color(img)
        #blue_mask, blue_masked_img = detect_blue_color(img)

        # 結果を出力
        #cv2.imwrite("C:\prog\python\\test\green_mask.png", green_mask)
        #cv2.imwrite("C:\prog\python\\test\green_masked_img.png", green_masked_img)

        while True:

            cv2.imshow('frame',green_mask)

            key_=cv2.waitKey(1)

            if key_==ord('a'):
                break

            if key_==ord('f'):
                
                fig_=plt.figure()

                axis1=fig_.add_subplot(1,2,1)
                axis2=fig_.add_subplot(1,2,2)

                axis1.imshow(green_mask,cmap='gray')
                axis2.imshow(green_masked_img,cmap='gray')

                plt.tight_layout()
                plt.show()





    elif key==ord('q'):

        ret__,threshold__=cv2.threshold(frame_hsv,0,256,cv2.THRESH_OTSU)
        print(ret__)

        plt.imshow(threshold__)
        plt.show()


       

    



        











