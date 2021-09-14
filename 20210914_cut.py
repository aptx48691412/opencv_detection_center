import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp


count=0

name_list=["asada","akiyoshi","abe","okami"]
sentence_list=["sentence_{}".format(i) for i in np.arange(1,4)]
count_list=["count_{}".format(i) for i in np.arange(1,4)]
situation_list=["None","situation"]

#cap=cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap=cv2.VideoCapture(0)


mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

# 動画ファイル保存用の設定
fps = int(cap.get(cv2.CAP_PROP_FPS)/2)                    # カメラのFPSを取得
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
#fps=30 
#w=640
#h=480            # カメラの縦幅を取得
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
video_list=list()
#video = cv2.VideoWriter("C:\Users\imdam\Desktop\opencv_exp\opencv_detection_center\data\{}_{}.mp4".format(), fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）
for i in name_list:
    for ii in situation_list:
        for iii in count_list:
            for iiii in sentence_list:
                video_list.append("{}_{}_{}_{}".format(i,ii,iii,iiii))  


def mediapipe_detection(frame):

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    results=hands.process(frame_rgb)
    ii=results.multi_hand_landmarks
    if ii:
        for i in ii:
            mpDraw.draw_landmarks(frame,i,mpHands.HAND_CONNECTIONS)

    gray_new=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    global threshold
    ret_,threshold=cv2.threshold(gray_new,0,255,cv2.THRESH_OTSU)

def get_center(img):

    try:
        y,x=np.where(img==255)
        x_avg,y_avg=np.average(x),np.average(y)

        return [int(x_avg),int(y_avg)]

    except:
        try:
            return [int(x_avg),int(y_avg)]
        except:
            None

def recording_movie():

    global count
    center_list=list()

    
    

    

    # 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
    while True:
        ret, frame = cap.read()       
        mediapipe_detection(frame)    
        lower_filter,higher_filter=np.array([224,224,224]),np.array([224,225,224])
        lower_filter,higher_filter=np.array([0,254,0]),np.array([0,255,0])
        
        inRange_mask=cv2.inRange(frame,lower_filter,higher_filter)
        bitwise=cv2.bitwise_and(frame,frame,mask=inRange_mask)
        bitwise_gray=cv2.cvtColor(bitwise,cv2.COLOR_BGR2GRAY)
        ret__,bitwise_threshold=cv2.threshold(bitwise_gray,0,255,cv2.THRESH_OTSU)
        center=get_center(bitwise_threshold)
        if center!=None:
            center_list.append(center)
            cv2.circle(bitwise,center,0,[150,255,30],25,200)

        video=cv2.VideoWriter("movie/{}.mp4".format(video_list[count]), fourcc, fps, (w, h)) # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

        video.write(bitwise) 
        cv2.imshow('frame',bitwise)
        key_=cv2.waitKey(1) 
        
        if key_==ord('c'):
            print(video_list[0])
            break 
        
        if key_==ord('p'):
            center_new_list=[i[0] for i in center_list]
            x=np.arange(0,len(center_new_list)*(1/fps),1/fps)
            
            fig=plt.figure()
            plt.plot(x,center_new_list,label='center')
            plt.grid()
            plt.xlabel('Time[sec]')
            plt.ylabel('center_placement')
            plt.title('center_placement_graph')
            plt.tight_layout()
            plt.legend()
            plt.show()

            #fig.savefig("C:/Users/imdam/Desktop/opencv_exp/opencv_detection_center/data/{}.png".format(str(video_list[count])))
            fig.savefig("img/{}.png".format(video_list[count]))




            count+=1


            break
                                # フレームを取得
                  



while True:

    ret,frame=cap.read()
    mediapipe_detection(frame)
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)

    if key==27:
        break

    elif key==ord('r'):
        recording_movie()
    

    elif key==ord('h'):
        while True:
            ret,frame=cap.read()
            mediapipe_detection(frame)
            cv2.imshow('frame',frame)
            key_=cv2.waitKey(1)

            if key_==ord('j'):
                break

            elif key_==ord('k'):
                frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                cv2.imwrite('/Users/yamauchisachiyo/Downloads/mediapipe.png',frame_hsv)


    elif key==ord('y'):

        #print(type(ii[0]))
        #print(len(ii))

        #print(type(ii[0]))
        #print(ii[0]['x'])

        print(ii)
        #print(len(ii))
        print(type(ii))
        print('--------------------')

        print(i)
        #print(len(i))
        print(type(i))
        print('--------------------')

        print(ii[0])
        print(type(ii[0]))

        print('--------------------')





        #print(ii[0][0])

        







    
    
