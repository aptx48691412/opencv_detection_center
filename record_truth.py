import pyaudio  #録音機能を使うためのライブラリ
import wave     #wavファイルを扱うためのライブラリ
import cv2

import numpy as np
import matplotlib.pyplot as plt
 
RECORD_SECONDS = 6 #録音する時間の長さ（秒）

count=0

name_list=["asada","akiyoshi","abe","okami"]
sentence_list=["sentence_{}".format(i) for i in np.arange(1,4)]
count_list=["count_{}".format(i) for i in np.arange(1,4)]
situation_list=["None","situation"]

WAVE_OUTPUT_FILENAME = list()
for i in name_list:
    for ii in situation_list:
        for iii in count_list:
            for iiii in sentence_list:
                WAVE_OUTPUT_FILENAME.append("{}_{}_{}_{}_".format(i,ii,iii,iiii)+"sample.wav")  #音声を保存するファイル名


iDeviceIndex = 0 #録音デバイスのインデックス番号
 
#基本情報の設定
FORMAT = pyaudio.paInt16 #音声のフォーマット
CHANNELS = 1             #モノラル
RATE = 44100             #サンプルレート
CHUNK = 2**11            #データ点数

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)

    if key==27:
        break


    if key==ord('r'):
        audio = pyaudio.PyAudio() #pyaudio.PyAudio()
 
        stream = audio.open(format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        input_device_index = iDeviceIndex, #録音デバイスのインデックス番号
        frames_per_buffer=CHUNK)
 



#--------------録音開始---------------
 
        print ("recording...")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
 
 
        print ("finished recording")
 
#--------------録音終了---------------
 
        stream.stop_stream()
        stream.close()
        audio.terminate()
 
        waveFile = wave.open('./sound/'+WAVE_OUTPUT_FILENAME[count], 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()



#///////

        data = b''.join(frames)

        


        


        x = np.frombuffer(data, dtype="int16") / 32768.0

        fig1=plt.figure(figsize=(15,3))
        plt.plot(x)
        plt.xlabel('[Hz]')
        plt.ylabel('volume')
        plt.grid()
        plt.title('sound_volume')
        plt.show()
        fig1.savefig('./sound/'+WAVE_OUTPUT_FILENAME[count]+'.png')

        x = np.fft.fft(np.frombuffer(data, dtype="int16"))

        fig2=plt.figure(figsize=(15,3))
        plt.plot(x.real[:int(len(x)/2)])
        plt.xlabel('[Hz]')
        plt.ylabel('volume')
        plt.grid()
        plt.title('sound_volume')
        plt.show()
        fig2.savefig('./sound/'+WAVE_OUTPUT_FILENAME[count]+'_FFT.png')

        count+=1
