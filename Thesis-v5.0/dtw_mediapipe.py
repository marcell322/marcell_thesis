#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
import numpy as np
from dtaidistance import dtw,dtw_ndim
from tqdm import tqdm, trange

# from scipy.spatial.distance import euclidean

# from fastdtw import fastdtw as dtw
from dtaidistance import dtw_visualisation as dtwvis
import pandas as pd
from tqdm import tqdm, trange
from pathlib import Path

IN_NAME = ['']
OUT_NAME = ['']


# In[ ]:


# opening file using tkinter

import tkinter as tk
from tkinter import filedialog, messagebox

def open_file_dialog(selected_file_label, FILE):

    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("Video files", "*.mp4"), ("All files", "*.*")])
    if file_path:
        selected_file_label.config(text=f"Selected File: {file_path}")
        FILE[0] = file_path

def Submit(root,input_filename,output_filename):
    if input_filename == '' or output_filename == '':
        messagebox.showerror("Attention","Need to have both student and teacher file")
    else:
        root.destroy()

root = tk.Tk()

input_file_label = tk.Label(root, text="Student File:")

open_button = tk.Button(root, text="Student File", command= lambda: open_file_dialog(input_file_label,IN_NAME))
open_button.pack(padx=20, pady=20)

input_file_label.pack()

ouput_file_label = tk.Label(root, text="Teacher File:")

open_button = tk.Button(root, text="Teacher File", command= lambda: open_file_dialog(ouput_file_label,OUT_NAME))
open_button.pack(padx=20, pady=20)

ouput_file_label.pack()

submit_button = tk.Button(root, text="Submit", command= lambda: Submit(root,IN_NAME[0],OUT_NAME[0]))
submit_button.pack(padx=20, pady=20)

root.mainloop()


# In[ ]:


IN_NAME = IN_NAME[0]
OUT_NAME = OUT_NAME[0]


# In[ ]:


# list of Constant
# IN_NAME = fr'..\media\Faded_pianella.mp4'
# IN_NAME = 'Faded.mp4'
# OUT_NAME = fr'..\media\output_roseau.mp4'
OUTMIX_NAME = fr'..\media\outputmix_roseau.mp4'
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

TWO_JOINTS = [ 
    [0,1], [1,2], [2,3], 
    [3,4], [0,5], [5,9], 
    [5,6], [6,7], [7,8], 
    [9,13], [9,10], [10,11], 
    [11,12], [13,17], [13,14], 
    [14,15], [15,16], [17,18], 
    [18,19], [19,20], [0,17]
]

TWO_JOINTS_NAME = [
    [
        'WRIST', 
        'THUMB_CMP'
    ],
    [
        'THUMB_CMP', 
        'THUMB_MCP'
    ],
    [
        'THUMB_MCP', 
        'THUMB_IP'
    ],
    [
        'THUMB_IP', 
        'THUMB_TIP'
    ],
    [
        'WRIST', 
        'INDEX_FINGER_MCP'
    ],
    [
        'INDEX_FINGER_MCP', 
        'INDEX_FINGER_PIP'
    ],
    [
        'INDEX_FINGER_PIP', 
        'INDEX_FINGER_DIP'
    ],
    [
        'INDEX_FINGER_DIP', 
        'INDEX_FINGER_TIP'
    ],
    [
        'INDEX_FINGER_MCP', 
        'MIDDLE_FINGER_MCP'
    ],
    [
        'MIDDLE_FINGER_MCP', 
        'MIDDLE_FINGER_PIP'
    ],
    [
        'MIDDLE_FINGER_PIP', 
        'MIDDLE_FINGER_DIP'
    ],
    [
        'MIDDLE_FINGER_DIP', 
        'MIDDLE_FINGER_TIP'
    ],
    [
        'MIDDLE_FINGER_MCP', 
        'RING_FINGER_MCP'
    ],
    [
        'RING_FINGER_MCP', 
        'RING_FINGER_PIP'
    ],
    [
        'RING_FINGER_PIP', 
        'RING_FINGER_DIP'
    ],
    [
        'RING_FINGER_DIP', 
        'RING_FINGER_TIP'
    ],
    [
        'RING_FINGER_MCP', 
        'PINKY_MCP'
    ],
    [
        'PINKY_MCP', 
        'PINKY_PIP'
    ],
    [
        'PINKY_PIP', 
        'PINKY_DIP'
    ],
    [
        'PINKY_DIP', 
        'PINKY_TIP'
    ],
    [
        'WRIST', 
        'PINKY_MCP'
    ],
]


# In[ ]:


#CREATING TEACHER EXCEL FILE

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(OUT_NAME)
wi = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fr = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
output_file = Path(OUT_NAME).stem
OUTPUT_PATH = fr'..\media\output_'+output_file+'.mp4'

out = cv2.VideoWriter(fr'..\media\output_'+output_file+'.mp4',fourcc, fr, (wi, hi))  
pose_tangan = [
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

alldata  = []
no_frame = []
frame_ctr = 0

fps = cap.get(cv2.CAP_PROP_FPS)
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
length = total_frame_count/fps

# pbar = tqdm(total = total_frame_count)
count = 0

from mediapipe.framework.formats import landmark_pb2
from tkinter import ttk

def process_video_with_progress_bar(cap, out, output_file, progress_bar, root):
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ctr = 0
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        print("Analyze teacher Video")
        print(total_frame_count)
        while cap.isOpened():
            success, image = cap.read()
            # pbar.update(frame_ctr)
            # count += fps*5 
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            frame_ctr += 1
            progress_bar['value'] = (frame_ctr / total_frame_count) * 100
            root.update_idletasks()

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape
            # print(len(image.shape))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # print("Hand ",f'{hand_landmarks.landmark}')
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    data_tangan  = {}

                    for i in range(len(pose_tangan)):
                        hand_landmarks.landmark[i].x = hand_landmarks.landmark[i].x * image.shape[0]
                        hand_landmarks.landmark[i].y = hand_landmarks.landmark[i].y * image.shape[1]
                        data_tangan.update(
                            {
                                pose_tangan[i] : f'{frame_ctr}'+", " +f'{hand_landmarks.landmark[i].x}' +", " +f'{hand_landmarks.landmark[i].y}'
                            }
                        )
                    alldata.append(data_tangan)
                    no_frame.append(frame_ctr)
            # Flip the image horizontally for a selfie-view display.
            frame = cv2.flip(image, 0)
            # image = cv2.resize(image, (960, 540)) 
            # cv2.rectangle(image, (20, 60), (120, 160), (0, 255, 0), 2)
            # cv2.imshow('MediaPipe Hands', image)

            out.write(image)  
            # print(frame_ctr)
            if (cv2.waitKey(5) & 0xFF == 27):
                break
        # print(alldata)
        print("Print Frame Data")
        df = pd.DataFrame(alldata)
        df.to_excel(fr'..\media\coordinate_'+output_file+'.xlsx')
        # df.to_excel("koordinat_faded_roseau.xlsx")
        cap.release()
        out.release()  
        cv2.destroyAllWindows()  
    root.quit()
    root.destroy()  # Destroy the Tkinter window

# Tkinter setup for progress bar
root = tk.Tk()
root.title("Video Processing with Progress Bar")

# Create a progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=20)

# Start video processing in a new thread to prevent freezing the UI
root.after(100, lambda: process_video_with_progress_bar(cap, out, output_file, progress_bar, root))

# Start the Tkinter event loop
root.mainloop()

# In[ ]:


#list of function
def finger_angle_3joints(hand_landmarks,index,joins_lists):
    # calculating finger angle
    angle_list = []
    for joint in joins_lists:
        a = np.array([hand_landmarks[index][joint[0]][0], hand_landmarks[index][joint[0]][1]]) # First coord
        b = np.array([hand_landmarks[index][joint[1]][0], hand_landmarks[index][joint[1]][1]]) # Second coord
        c = np.array([hand_landmarks[index][joint[2]][0], hand_landmarks[index][joint[2]][1]]) # Third coord
        
        radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        angle_list.append(angle)

    return angle_list

def finger_angle_2joints(hand_landmarks,index,joins_lists):
    # calculating finger angle
    angle_list = []
    # print(len(hand_landmarks))
    for joint in joins_lists:
        a = np.array([hand_landmarks[index][joint[0]][0], hand_landmarks[index][joint[0]][1]]) # First coord
        b = np.array([hand_landmarks[index][joint[1]][0], hand_landmarks[index][joint[1]][1]]) # Second coord
        
        radians = np.arctan2(b[1] - a[1],b[0]-a[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        angle_list.append(angle)

    return angle_list


# In[ ]:


# Configuration
FPS = 30
FFT_WINDOW_SECONDS = 0.25 # how many seconds of audio make up an FFT window

# Note range to display
FREQ_MIN = 10
FREQ_MAX = 1000

# Notes to display
TOP_NOTES = 3

# Names of the notes
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Output size. Generally use SCALE for higher res, unless you need a non-standard aspect ratio.
RESOLUTION = (1920, 1080)
SCALE = 2 # 0.5=QHD(960x540), 1=HD(1920x1080), 2=4K(3840x2160)


# In[ ]:


import plotly.graph_objects as go
import numpy as np

def freq_to_number(f): return 69 + 12*np.log2(f/440.0)
def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)
def note_name(n): 
  global NOTE_NAMES
  return NOTE_NAMES[n % 12] + str(int(n/12 - 1))

def plot_fft(p, xf, fs, notes, dimensions=(960,540)):
  global FREQ_MAX
  global FREQ_MIN
  layout = go.Layout(
      title="frequency spectrum",
      autosize=False,
      width=dimensions[0],
      height=dimensions[1],
      xaxis_title="Frequency (note)",
      yaxis_title="Magnitude",
      font={'size' : 24}
  )

  fig = go.Figure(layout=layout,
                  layout_xaxis_range=[FREQ_MIN,FREQ_MAX],
                  layout_yaxis_range=[0,1]
                  )
  
  fig.add_trace(go.Scatter(
      x = xf,
      y = p))
  
  for note in notes:
    fig.add_annotation(x=note[0]+10, y=note[2],
            text=note[1],
            font = {'size' : 48},
            showarrow=False)
  return fig

def extract_sample(audio, frame_number, frame_offset):
  global FFT_WINDOW_SIZE
  end = frame_number * frame_offset
  begin = int(end - FFT_WINDOW_SIZE)

  if end == 0:
    # We have no audio yet, return all zeros (very beginning)
    return np.zeros((np.abs(begin)),dtype=float)
  elif begin<0:
    # We have some audio, padd with zeros
    return np.concatenate([np.zeros((np.abs(begin)),dtype=float),audio[0:end]])
  else:
    # Usually this happens, return the next sample
    return audio[begin:end]

def find_top_notes(fft,num, xf):
  if np.max(fft.real)<0.001:
    return []

  lst = [x for x in enumerate(fft.real)]
  lst = sorted(lst, key=lambda x: x[1],reverse=True)

  idx = 0
  found = []
  found_note = set()
  while( (idx<len(lst)) and (len(found)<num) ):
    f = xf[lst[idx][0]]
    y = lst[idx][1]
    n = freq_to_number(f)
    n0 = int(round(n))
    name = note_name(n0)

    if name not in found_note:
      found_note.add(name)
      s = [f,note_name(n0),y]
      found.append(s)
    idx += 1
    
  return found


# In[ ]:


koordinat_guru = []
guru = []

def open_teacher_file(output_file, progress_bar, root):
    #teaching file calculation
    global koordinat_guru
    global guru
    print("opening teacher file")
    teacher = pd.read_excel(fr'..\media\coordinate_'+output_file+'.xlsx')
    # for i in tqdm(range(len(teacher['WRIST']))):
    for i in (range(len(teacher['WRIST']))):
        # frame.append([])
        koordinat_guru.append([])
        progress_bar['value'] = (i / len(teacher['WRIST'])) * 100
        root.update_idletasks()
        for pose in (POSE_TANGAN):
            temp = (teacher[pose][i].split(', '))
            koordinat_guru[i].append([eval(temp[1]),eval(temp[2])])
            # koordinat_guru[i].append()
        guru.append(finger_angle_2joints(koordinat_guru,i,TWO_JOINTS))

    guru = np.array(guru)
    root.quit()
    root.destroy()  # Destroy the Tkinter window

# Tkinter setup for progress bar
root = tk.Tk()
root.title("Loading Teacher Value")

# Create a progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=20)

# Start video processing in a new thread to prevent freezing the UI
root.after(100, lambda: open_teacher_file(output_file,progress_bar, root))

# Start the Tkinter event loop
root.mainloop()
# print(teacher['WRIST'][0])
# print(koordinat_guru[0])
# print(guru)


# In[ ]:


# Buka buka file dan setting variable
cap = cv2.VideoCapture(IN_NAME)
cap_teach = cv2.VideoCapture(OUTPUT_PATH)

# loading video dsa gfg intro video 
# clip = VideoFileClip(IN_NAME) 
  
# getting audio from the clip 
# audio_clip = clip.audio

wi = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fr = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
# out = cv2.VideoWriter(IN_NAME,fourcc, fr, (wi, hi))  

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


alldata  = []
list_d = []
frame_ctr = 0


# In[ ]:


#list of function
def finger_angle_3joints(hand_landmarks,index,joins_lists):
    # calculating finger angle
    angle_list = []
    for joint in joins_lists:
        a = np.array([hand_landmarks[index][joint[0]][0], hand_landmarks[index][joint[0]][1]]) # First coord
        b = np.array([hand_landmarks[index][joint[1]][0], hand_landmarks[index][joint[1]][1]]) # Second coord
        c = np.array([hand_landmarks[index][joint[2]][0], hand_landmarks[index][joint[2]][1]]) # Third coord
        
        radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        angle_list.append(angle)

    return angle_list

def finger_angle_2joints(hand_landmarks,index,joins_lists):
    # calculating finger angle
    angle_list = []
    # print(len(hand_landmarks))
    # print(index)
    for joint in joins_lists:
        a = np.array([hand_landmarks[index][joint[0]][0], hand_landmarks[index][joint[0]][1]]) # First coord
        b = np.array([hand_landmarks[index][joint[1]][0], hand_landmarks[index][joint[1]][1]]) # Second coord
        
        radians = np.arctan2(b[1] - a[1],b[0]-a[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        angle_list.append(angle)

    return angle_list


# In[ ]:

print("Showing both video!")

# Define font and other properties for the text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)  # White color for the text
thickness = 2
position1 = (50, 50)  # Position for the text in the first video
position2 = (50, 50)  # Position for the text in the second video


fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Example codec
# out_final = cv2.VideoWriter(fr'..\media\output_final.avi',fourcc, 20.0, (720, 720))
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_ctr = 0

def process_final_video_with_progress_bar(cap, out_final):
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ctr = 0
    with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
        with ThreadPoolExecutor() as executor:
            # while tqdm(cap.isOpened()):
            while (cap.isOpened()):
                success, image = cap.read()
                success_teach, image_teach = cap_teach.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    print(frame_ctr)
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                if not success_teach:
                    # print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                frame_ctr += 1
                # progress_bar['value'] = (frame_ctr / total_frame_count) * 100
                # root.update_idletasks()

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

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image_teach.flags.writeable = False
                image_teach = cv2.cvtColor(image_teach, cv2.COLOR_BGR2RGB)
                results_teach = hands.process(image_teach)
                
                # Draw the hand annotations on the image.
                image_teach.flags.writeable = True
                image_teach = cv2.cvtColor(image_teach, cv2.COLOR_RGB2BGR)
                image_teach_height, image_teach_width, _ = image_teach.shape
                # for dtw
                if len(alldata) > 0:
                    # for i in range(0, len(alldata) - WINDOW_SIZE + 1, STRIDE):
                    i = ((len(alldata) - 1) // WINDOWING_SIZE) * WINDOWING_SIZE
                    df = pd.DataFrame(alldata[i:i + WINDOWING_SIZE])
                    koordinat_murid = []
                    murid = []
                    for j in range(0, len(df['WRIST'])):
                        koordinat_murid.append([])
                        for pose in POSE_TANGAN:
                            temp = (df[pose][j].split(', '))
                            koordinat_murid[j].append([eval(temp[1]),eval(temp[2])])
                            # koordinat_murid[j].append()
                        murid.append(finger_angle_2joints(koordinat_murid,j,TWO_JOINTS))
                    murid = np.array(murid)
                
                    # dtw calculate distance with stride and window
                    windowed_video_landmarks = murid
                    windowed_excel_landmarks = guru[i:i + WINDOWING_SIZE]
                    d = dtw_ndim.distance(windowed_excel_landmarks, windowed_video_landmarks, window=WINDOW_SIZE)
                    list_d.append(d)
                    # d = sum(list_d) / len(list_d)
                    spit = 100 * 1.07 * np.exp(-0.17 * np.average(list_d))
                    import math
                    spit = math.floor(spit * 10000) / 10000

                    text_score = "Score: "+str(spit)
                    
                    # Get frame dimensions for bottom-right text placement
                    frame1_height, frame1_width = image.shape[:2]

                    # Calculate the size of the text to make sure it fits
                    text_size, _ = cv2.getTextSize(text_score, font, font_scale, thickness)

                    # Set the position for the bottom-right corner of the top video
                    text_x = frame1_width - text_size[0] - 10  # 10 pixels padding from the right
                    text_y = frame1_height - 10  # 10 pixels padding from the bottom


                    # Add text at the bottom-right corner of the top video (Video 1)
                    cv2.putText(image, text_score, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

                    # print("distance: "+ str(d))
                    
                # Flip the image horizontally for a selfie-view display.
                frame = cv2.flip(image, 0)
                image = cv2.resize(image, (720, 360)) 
                image_teach = cv2.resize(image_teach, (720, 360)) 
                cv2.putText(image, 'Student', position1, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(image_teach, 'Teacher', position2, font, font_scale, color, thickness, cv2.LINE_AA)


                canvas = cv2.vconcat([image, image_teach])

                cv2.imshow('MediaPipe Hands', canvas)
                
                # out_final.write(canvas)  

                if (cv2.waitKey(5) & 0xFF == 27):
                    break
    print("Canvas size:", canvas.shape)
    print("Canvas type:", canvas.dtype)
    cap.release()
    # out_final.release()  
    cv2.destroyAllWindows() 
    # root.quit()
    # root.destroy()  # Destroy the Tkinter window

# Tkinter setup for progress bar
# root = tk.Tk()
# root.title("Video Processing Final output with Progress Bar")

# # Create a progress bar
# progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
# progress_bar.pack(pady=20)

# # Start video processing in a new thread to prevent freezing the UI
# root.after(100, lambda: process_final_video_with_progress_bar(cap, out, output_file, progress_bar, root))

# # Start the Tkinter event loop
# root.mainloop()
process_final_video_with_progress_bar(cap, '')


# In[ ]:


# result
# print("List")
# print(list_d)


spit = 100 * 1.07 * np.exp(-0.17 * np.average(list_d))
print("Score: "+ str(spit))

def Close(root):
    root.destroy()

root = tk.Tk()

score_label = tk.Label(root, text="Score:" + str(spit))
score_label.pack()

close_button = tk.Button(root, text="Close", command= lambda: Close(root))
close_button.pack(padx=20, pady=20)

root.mainloop()