import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp


cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

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

    
    # 動画ファイル保存用の設定
    fps = int(cap.get(cv2.CAP_PROP_FPS)/2)                    # カメラのFPSを取得
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))             # カメラの縦幅を取得
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
    video = cv2.VideoWriter("/Users/yamauchisachiyo/Downloads/test_dfghdyky.mp4", fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

    center_list=list()

    # 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
    while True:
        ret, frame = cap.read()       
        mediapipe_detection(frame)    
        lower_filter,higher_filter=np.array([0,254,0]),np.array([0,255,0])
        inRange_mask=cv2.inRange(frame,lower_filter,higher_filter)
        bitwise=cv2.bitwise_and(frame,frame,mask=inRange_mask)
        bitwise_gray=cv2.cvtColor(bitwise,cv2.COLOR_BGR2GRAY)
        ret__,bitwise_threshold=cv2.threshold(bitwise_gray,0,255,cv2.THRESH_OTSU)
        center=get_center(bitwise_threshold)
        if center!=None:
            center_list.append(center)
            cv2.circle(bitwise,center,0,[150,255,30],25,200)

        video.write(bitwise) 
        cv2.imshow('frame',bitwise)
        key_=cv2.waitKey(1) 
        
        if key_==ord('c'):
            break 
        
        if key_==ord('p'):
            center_new_list=[i[0] for i in center_list]
            x=np.arange(0,len(center_new_list)*(1/fps),1/fps)
            plt.plot(x,center_new_list,label='center')
            plt.grid()
            plt.xlabel('Time[sec]')
            plt.ylabel('center_placement')
            plt.title('center_placement_graph')
            plt.tight_layout()
            plt.legend()
            plt.show()


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

        







    
