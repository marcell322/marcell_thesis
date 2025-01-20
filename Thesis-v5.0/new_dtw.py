import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from moviepy import *
import tkinter as tk
from tkinter import filedialog, messagebox
import mediapipe as mp
from pathlib import Path
import sounddevice as sd
import ffmpeg

STUDENT_NAME = ['']
TEACHER_NAME = ['']

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

open_button = tk.Button(root, text="Student File", command= lambda: open_file_dialog(input_file_label,STUDENT_NAME))
open_button.pack(padx=20, pady=20)

input_file_label.pack()

ouput_file_label = tk.Label(root, text="Teacher File:")

open_button = tk.Button(root, text="Teacher File", command= lambda: open_file_dialog(ouput_file_label,TEACHER_NAME))
open_button.pack(padx=20, pady=20)

ouput_file_label.pack()

submit_button = tk.Button(root, text="Submit", command= lambda: Submit(root,STUDENT_NAME[0],TEACHER_NAME[0]))
submit_button.pack(padx=20, pady=20)

root.mainloop()


STUDENT_NAME = STUDENT_NAME[0]
TEACHER_NAME = TEACHER_NAME[0]
WINDOW_SIZE = 1
WINDOWING_SIZE = 5
STRIDE = 3

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

# Video Teacher to excel
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(TEACHER_NAME)
wi = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fr = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
teacher_file = Path(TEACHER_NAME).stem
TEACHER_PATH = fr'..\media\teacher_'+teacher_file+'.mp4'

out = cv2.VideoWriter(fr'..\media\teacher_'+teacher_file+'.mp4',fourcc, fr, (wi, hi))  
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

def extract_audio(input_video, output_audio):
  """
  Extracts the audio from a video file using ffmpeg.

  Args:
    input_video: Path to the input video file.
    output_audio: Path to the output audio file.
  """
  try:
    ffmpeg.input(input_video).output(output_audio).overwrite_output().run()
    print(f"Audio extracted successfully from {input_video} to {output_audio}")
  except ffmpeg.Error as e:
    print(f"An error occurred during audio extraction: {e}")

# Audio Teacher to excel
video = VideoFileClip(TEACHER_NAME)
audio = video.audio
audio_file = fr'..\media\teacher_'+teacher_file+'_audio.wav'
extract_audio(TEACHER_NAME,audio_file)

y, sr = librosa.load(audio_file, sr=None)

def play_audio(audio_data, sample_rate):
    """Plays audio data in the background."""
    sd.play(audio_data, samplerate=sample_rate)

# Start playing audio
play_audio(y, sr)

# Step 5: Show video frames synchronized with audio
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
delay = 1 / frame_rate

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display video frame
    # ax_video.clear()
    # ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # ax_video.set_title("Video Frame")
    # ax_video.axis("off")

    # plt.pause(delay)
    cv2.imshow('',frame)
    delay = (int)(100 / fps)
    key = cv2.waitKey(delay)
    if (cv2.waitKey(5) & 0xFF == 27):
        break

# Release resources
cap.release()
cv2.destroyAllWindows() 
sd.stop()  # Stop audio playback
