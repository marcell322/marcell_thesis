import cv2
from moviepy.editor import *
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
import numpy as np
from dtaidistance import dtw,dtw_ndim
from tqdm import tqdm

# from scipy.spatial.distance import euclidean

# from fastdtw import fastdtw as dtw
from dtaidistance import dtw_visualisation as dtwvis
import pandas as pd
import math
import tkinter as tk
from tkinter import filedialog

def finger_angle(hand_landmarks,index,joins_lists):
    # calculating finger angle
    angle_list = []
    for joint in joins_lists:
        a = np.array([hand_landmarks[joint[0]][index][0], hand_landmarks[joint[0]][index][1]]) # First coord
        b = np.array([hand_landmarks[joint[1]][index][0], hand_landmarks[joint[1]][index][1]]) # Second coord
        
        rad = np.arctan2(a[1]-b[1], a[0]-b[0])

        angle = rad * (180 / np.pi)

        if angle > 180.0:
            angle = 360-angle
        angle_list.append(angle)

    return angle_list

IN_NAME = 'Faded_pianella.mp4'
# IN_NAME = 'Faded.mp4'
OUT_NAME = 'output_roseau.mp4'
OUTMIX_NAME = 'outputmix_roseau.mp4'
WINDOW_SIZE = 1
WINDOWING_SIZE = 5
STRIDE = 3
POSE_TANGAN = [
    'WRIST', 
    'THUMB_CMP', 
    'THUMB_MCP', 
    'THUMB_IP', 
    'THUMB_TIP', 
    'INDEX_FINGER_MCP', 
    'INDEX_FINGER_PIP', 
    'INDEX_FINGER_DIP', 
    'INDEX_FINGER_TIP', 
    'MIDDLE_FINGER_MCP',
    'MIDDLE_FINGER_PIP', 
    'MIDDLE_FINGER_DIP', 
    'MIDDLE_FINGER_TIP', 
    'RING_FINGER_MCP', 
    'RING_FINGER_PIP', 
    'RING_FINGER_DIP',
    'RING_FINGER_TIP', 
    'PINKY_MCP', 
    'PINKY_PIP', 
    'PINKY_DIP', 
    'PINKY_TIP'
]
JOINT_LIST = [[2,3,4] , [8,7,6], [12,11,10], [16,15,14], [20,19,18]]
JOINT_LIST_NAME = [
    [
        'THUMB_MCP', 
        'THUMB_IP', 
    ],
    [
        'THUMB_IP', 
        'THUMB_TIP'
    ],
    [
        'INDEX_FINGER_PIP', 
        'INDEX_FINGER_DIP', 
        'INDEX_FINGER_TIP'
    ],
    [
        'MIDDLE_FINGER_PIP', 
        'MIDDLE_FINGER_DIP', 
        'MIDDLE_FINGER_TIP'
    ],
    [ 
        'RING_FINGER_PIP', 
        'RING_FINGER_DIP',
        'RING_FINGER_TIP'
    ],
    [ 
        'PINKY_PIP', 
        'PINKY_DIP', 
        'PINKY_TIP'
    ]
]

# open the teacher file to teach
print("opening teacher file")
teacher = pd.read_excel('koordinat_faded_roseau.xlsx')
koordinat_guru = []
guru = []
for i in tqdm(range(len(teacher['WRIST']))):
    # frame.append([])
    koordinat_guru.append([])
    for pose in (POSE_TANGAN):
        temp = (teacher[pose][i].split(', '))
        koordinat_guru[i].append([eval(temp[1]),eval(temp[2])])
        # koordinat_guru[i].append()
    guru.append(finger_angle(koordinat_guru,i,JOINT_LIST))

koordinat_guru = np.array(koordinat_guru)

# print(teacher['WRIST'][0])
# print(koordinat_guru[0])
print(guru)
exit()
print("Finish")

cap = cv2.VideoCapture(IN_NAME)

# loading video dsa gfg intro video 
# clip = VideoFileClip(IN_NAME) 
  
# getting audio from the clip 
# audio_clip = clip.audio

wi = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fr = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter(OUT_NAME,fourcc, fr, (wi, hi))  

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


alldata  = []
list_d = []
frame_ctr = 0


print("processing mediapipe")
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    with ThreadPoolExecutor() as executor:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            frame_ctr += 1
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # saving hand coordinate
                    data_tangan  = {}
                    for i in range(len(POSE_TANGAN)):
                        hand_landmarks.landmark[i].x = hand_landmarks.landmark[i].x * image.shape[0]
                        hand_landmarks.landmark[i].y = hand_landmarks.landmark[i].y * image.shape[1]
                        data_tangan.update(
                            {
                                POSE_TANGAN[i] : f'{frame_ctr}'+", " +f'{hand_landmarks.landmark[i].x}' +", " +f'{hand_landmarks.landmark[i].y}'
                            }
                        )
                    alldata.append(data_tangan)

            # for dtw
            if len(alldata) > 0:
                # for i in range(0, len(alldata) - WINDOW_SIZE + 1, STRIDE):
                i = ((len(alldata) - 1) // WINDOWING_SIZE) * WINDOWING_SIZE
                df = pd.DataFrame(alldata[i:i + WINDOWING_SIZE])
                koordinat_murid = []
                for j in range(0, len(df['WRIST'])):
                    koordinat_murid.append([])
                    for pose in POSE_TANGAN:
                        temp = (df[pose][j].split(', '))
                        koordinat_murid[j].append([eval(temp[1]),eval(temp[2])])
                        # koordinat_murid[j].append()
                koordinat_murid = np.array(koordinat_murid)
            
                # dtw calculate distance with stride and window
                windowed_video_landmarks = koordinat_murid
                windowed_excel_landmarks = koordinat_guru[i:i + WINDOWING_SIZE]
                d = dtw_ndim.distance(windowed_excel_landmarks, windowed_video_landmarks, window=WINDOW_SIZE)
                list_d.append(d)
                # d = sum(list_d) / len(list_d)

                # print("distance: "+ str(d))
                
            # Flip the image horizontally for a selfie-view display.
            frame = cv2.flip(image, 0)
            image = cv2.resize(image, (960, 540)) 
            # cv2.imshow('MediaPipe Hands', image)

            
            # out.write(image)  
            if (cv2.waitKey(5) & 0xFF == 27):
                break

cap.release()
# out.release()  
cv2.destroyAllWindows() 

print(list_d)

# print("saving audio")
# video_clip = VideoFileClip(OUT_NAME)

# # video_clip.set_audio(audio_clip)

# # video_clip.write_videofile(OUTMIX_NAME,fps=30, audio_codec="aac", audio_bitrate="192k")

# new_audioclip = CompositeAudioClip([audio_clip])
# video_clip.audio = new_audioclip
# video_clip.write_videofile(OUTMIX_NAME)