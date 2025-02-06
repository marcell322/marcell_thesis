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
from tkinter import ttk
import pandas as pd
from dtaidistance import dtw,dtw_ndim
import math

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
HAND_POSE = [
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
    for joint in joins_lists:
        a = np.array([hand_landmarks[index][joint[0]][0], hand_landmarks[index][joint[0]][1]]) # First coord
        b = np.array([hand_landmarks[index][joint[1]][0], hand_landmarks[index][joint[1]][1]]) # Second coord
        
        radians = np.arctan2(b[1] - a[1],b[0]-a[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        angle_list.append(angle)

    return angle_list


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

def play_audio(audio_data, sample_rate):
    """Plays audio data in the background."""
    sd.play(audio_data, samplerate=sample_rate)

def get_min_freq(song,sr):
    # Calculate the Short-Time Fourier Transform (STFT)
    D = librosa.stft(song)

    # Convert the STFT to magnitude 
    S = np.abs(D)
    # Get the frequencies corresponding to the STFT bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0])

    positive_freqs = freqs[freqs > 0]
    # Find the index of the minimum frequency
    min_freq_idx = np.argmin(positive_freqs)

    # Get the minimum frequency value
    min_freq = positive_freqs[min_freq_idx]
    return min_freq

def get_midi_seq(cqt_magnitude,midi_numbers):
    midi_sequence = []
    for frames in cqt_magnitude.T:  # Iterate over each time frame
        max_bin = np.argmax(frames)  # Find the bin with the highest energy
        midi_note = round(midi_numbers[max_bin])  # Round to the nearest MIDI note
        midi_sequence.append(midi_note)
    return midi_sequence

def assign_cqt_to_frames(my_sr,my_fps,my_tot_frames, my_tot_seq,my_midi_seq):
    samples_per_frame = my_sr / my_fps
    cqt_sequences_per_frame = my_tot_seq // my_tot_frames

    # Initialize a list to store CQT sequence indices for each frame
    frame_cqt_indices = [[] for _ in range(my_tot_frames)]

    # Assign CQT sequences to frames
    cqt_index = 0
    for frame_idx in range(my_tot_frames):
        frame_cqt_indices[frame_idx] = (my_midi_seq[cqt_index])
        cqt_index += cqt_sequences_per_frame 
    
    frame_cqt_indices = np.array(frame_cqt_indices)

    return frame_cqt_indices

def process_video_with_progress_bar(cap, out, output_file, progress_bar, root):
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ctr = 0
    alldata = []
    no_frame = []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        print("Analyze Video")
        while cap.isOpened():
            success, image = cap.read()
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
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    data_tangan  = {}

                    for i in range(len(HAND_POSE)):
                        hand_landmarks.landmark[i].x = hand_landmarks.landmark[i].x * image.shape[0]
                        hand_landmarks.landmark[i].y = hand_landmarks.landmark[i].y * image.shape[1]
                        data_tangan.update(
                            {
                                HAND_POSE[i] : f'{frame_ctr}'+", " +f'{hand_landmarks.landmark[i].x}' +", " +f'{hand_landmarks.landmark[i].y}'
                            }
                        )
                    alldata.append(data_tangan)
                    no_frame.append(frame_ctr)
            # Flip the image horizontally for a selfie-view display.
            frame = cv2.flip(image, 0)

            out.write(image)  
            if (cv2.waitKey(5) & 0xFF == 27):
                break
            
        print("Print Frame Data")
        df = pd.DataFrame(alldata)
        df.to_excel(fr'..\media\coordinate_'+output_file+'.xlsx')
        cap.release()
        out.release()  
        cv2.destroyAllWindows()  
    root.quit()
    root.destroy()  # Destroy the Tkinter window


def open_excel_teacher(output_file, progress_bar, root):
    #teaching file calculation
    global teacher_coordinate
    global teacher_angle
    print("opening teacher file")
    teacher_excel = pd.read_excel(fr'..\media\coordinate_'+output_file+'.xlsx')
    # for i in tqdm(range(len(teacher['WRIST']))):
    for i in (range(len(teacher_excel['WRIST']))):
        # frame.append([])
        teacher_coordinate.append([])
        progress_bar['value'] = (i / len(teacher_excel['WRIST'])) * 100
        root.update_idletasks()
        for pose in (HAND_POSE):
            temp = (teacher_excel[pose][i].split(', '))
            teacher_coordinate[i].append([eval(temp[1]),eval(temp[2])])
        teacher_angle.append(finger_angle_2joints(teacher_coordinate,i,TWO_JOINTS))

    teacher_angle = np.array(teacher_angle)
    root.quit()
    root.destroy()  # Destroy the Tkinter window
    
def open_excel_student(output_file, progress_bar, root):
    #teaching file calculation
    global student_coordinate
    global student_angle
    print("opening student file")
    student_excel = pd.read_excel(fr'..\media\coordinate_'+output_file+'.xlsx')
    # for i in tqdm(range(len(student['WRIST']))):
    for i in (range(len(student_excel['WRIST']))):
        # frame.append([])
        student_coordinate.append([])
        progress_bar['value'] = (i / len(student_excel['WRIST'])) * 100
        root.update_idletasks()
        for pose in (HAND_POSE):
            temp = (student_excel[pose][i].split(', '))
            student_coordinate[i].append([eval(temp[1]),eval(temp[2])])
        student_angle.append(finger_angle_2joints(student_coordinate,i,TWO_JOINTS))

    student_angle = np.array(student_angle)
    root.quit()
    root.destroy()  # Destroy the Tkinter window

# Video Teacher to excel
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

teacher_cap = cv2.VideoCapture(TEACHER_NAME)
teacher_wi = int(teacher_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
teacher_hi = int(teacher_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
teacher_fr = int(teacher_cap.get(cv2.CAP_PROP_FPS))
teacher_frame = int(teacher_cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
teacher_file = Path(TEACHER_NAME).stem
TEACHER_PATH = fr'..\media\mp_'+teacher_file+'.mp4'

teacher_out = cv2.VideoWriter(TEACHER_PATH,fourcc, teacher_fr, (teacher_wi, teacher_hi))  

# Audio Teacher to excel
teacher_video = VideoFileClip(TEACHER_NAME)
teacher_audio = teacher_video.audio
teacher_audio_file = fr'..\media\audio_'+teacher_file+'_audio.wav'
extract_audio(TEACHER_NAME,teacher_audio_file)
teacher_video.close()

teacher_y, teacher_sr = librosa.load(teacher_audio_file, sr=None)

teacher_min_freq = get_min_freq(teacher_y,teacher_sr)
TFMIN = teacher_min_freq
BINS_PER_OCTAVE = 12

# Compute the Constant-Q Transform
teacher_cqt = librosa.cqt(teacher_y, sr=teacher_sr, fmin=TFMIN, bins_per_octave=BINS_PER_OCTAVE)

teacher_n_bins = teacher_cqt.shape[0]
teacher_frequencies = TFMIN * (2.0 ** (np.arange(teacher_n_bins) / BINS_PER_OCTAVE))

# Convert frequencies to MIDI note numbers
teacher_midi_numbers = 69 + 12 * np.log2(teacher_frequencies / 440.0)


# Get the sequence of MIDI values
# For each time frame, find the bin with the maximum amplitude
teacher_cqt_magnitude = np.abs(teacher_cqt)  # Get the magnitude of the CQT
teacher_midi_sequence = get_midi_seq(teacher_cqt_magnitude,teacher_midi_numbers)

teacher_total_cqt_sequences = teacher_cqt_magnitude.T.shape[0]

teacher_cqt_to_frames = assign_cqt_to_frames(teacher_sr,teacher_fr,teacher_frame,teacher_total_cqt_sequences,teacher_midi_sequence)

# Tkinter setup for progress bar
root = tk.Tk()
root.title("Video Processing with Progress Bar")

# Create a progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=20)

# Start video processing in a new thread to prevent freezing the UI
root.after(100, lambda: process_video_with_progress_bar(teacher_cap, teacher_out, teacher_file, progress_bar, root))

# Loading teacher coordinate & angle
teacher_coordinate = []
teacher_angle = []
# Start the Tkinter event loop
root.mainloop()

# Tkinter setup for progress bar
root = tk.Tk()
root.title("Loading Teacher Value")

# Create a progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=20)

# Start video processing in a new thread to prevent freezing the UI
root.after(100, lambda: open_excel_teacher(teacher_file, progress_bar, root))

# Start the Tkinter event loop
root.mainloop()

# start detecting student
student_cap = cv2.VideoCapture(STUDENT_NAME)
student_wi = int(student_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
student_hi = int(student_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
student_fr = int(student_cap.get(cv2.CAP_PROP_FPS))
student_frame = int(student_cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
student_file = Path(STUDENT_NAME).stem
STUDENT_PATH = fr'..\media\mp_'+student_file+'.mp4'

student_out = cv2.VideoWriter(STUDENT_PATH,fourcc, student_fr, (student_wi, student_hi))  

# Audio Student to excel
student_video = VideoFileClip(STUDENT_NAME)
student_audio = student_video.audio
student_audio_file = fr'..\media\audio_'+student_file+'_audio.wav'
extract_audio(STUDENT_NAME,student_audio_file)
student_video.close()

student_y, student_sr = librosa.load(student_audio_file, sr=None)

student_min_freq = get_min_freq(student_y,student_sr)
TFMIN = student_min_freq
BINS_PER_OCTAVE = 12

# Compute the Constant-Q Transform
student_cqt = librosa.cqt(student_y, sr=student_sr, fmin=TFMIN, bins_per_octave=BINS_PER_OCTAVE)

student_n_bins = student_cqt.shape[0]
student_frequencies = TFMIN * (2.0 ** (np.arange(student_n_bins) / BINS_PER_OCTAVE))

# Convert frequencies to MIDI note numbers
student_midi_numbers = 69 + 12 * np.log2(student_frequencies / 440.0)


# Get the sequence of MIDI values
# For each time frame, find the bin with the maximum amplitude
student_cqt_magnitude = np.abs(student_cqt)  # Get the magnitude of the CQT
student_midi_sequence = get_midi_seq(student_cqt_magnitude,student_midi_numbers)

student_total_cqt_sequences = student_cqt_magnitude.T.shape[0]

student_cqt_to_frames = assign_cqt_to_frames(student_sr,student_fr,student_frame,student_total_cqt_sequences,student_midi_sequence)

# Tkinter setup for progress bar
root = tk.Tk()
root.title("Video Processing with Progress Bar")

# Create a progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=20)

# Start video processing in a new thread to prevent freezing the UI
root.after(100, lambda: process_video_with_progress_bar(student_cap, student_out, student_file, progress_bar, root))

# Loading student coordinate & angle
student_coordinate = []
student_angle = []
# Start the Tkinter event loop
root.mainloop()

# Tkinter setup for progress bar
root = tk.Tk()
root.title("Loading Teacher Value")

# Create a progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=20)

# Start video processing in a new thread to prevent freezing the UI
root.after(100, lambda: open_excel_student(student_file, progress_bar, root))

# Start the Tkinter event loop
root.mainloop()

list_d_landmark = []
list_d_audio = []
list_d_concat = []

list_score_landmark = []
list_score_audio = []
list_score_concat = []

student_angle_normalized = (student_angle - student_angle.min()) / (student_angle.max() - student_angle.min())
teacher_angle_normalized = (teacher_angle - teacher_angle.min()) / (teacher_angle.max() - teacher_angle.min())

student_audio_normalized = (student_cqt_to_frames - student_cqt_to_frames.min()) / (student_cqt_to_frames.max() - student_cqt_to_frames.min())
teacher_audio_normalized = (teacher_cqt_to_frames - teacher_cqt_to_frames.min()) / (teacher_cqt_to_frames.max() - teacher_cqt_to_frames.min())
for i in range(len(student_audio_normalized)):
    j = ((i - 1) // WINDOWING_SIZE) * WINDOWING_SIZE
    windowed_student_landmarks = student_angle_normalized[j:j + WINDOWING_SIZE]
    windowed_teacher_landmarks = teacher_angle_normalized[j:j + WINDOWING_SIZE]
    windowed_student_cqt = student_audio_normalized[j:j + WINDOWING_SIZE]
    windowed_teacher_cqt = teacher_audio_normalized[j:j + WINDOWING_SIZE]
    d_landmark = dtw_ndim.distance(windowed_student_landmarks, windowed_teacher_landmarks, window=WINDOW_SIZE)
    d_audio = dtw_ndim.distance(windowed_student_cqt,windowed_teacher_cqt, window=WINDOW_SIZE)
    d_audio = dtw_ndim.distance(windowed_teacher_cqt, windowed_student_cqt, window=WINDOW_SIZE)
    list_d_landmark.append(d_landmark)
    list_d_audio.append(d_audio)
    spit_landmark = 100 * 1.07 * np.exp(-0.17 * np.average(list_d_landmark))
    spit_audio = 100 * 1.07 * np.exp(-0.17 * np.average(list_d_audio))
    spit_concat = spit_audio * 0.5 + spit_landmark * 0.5
    list_score_landmark.append(spit_landmark)
    list_score_audio.append(spit_audio)
    list_score_concat.append(spit_concat)
    
    
spit_landmark = 100 * 1.07 * np.exp(-0.17 * np.average(list_d_landmark))
spit_audio = 100 * 1.07 * np.exp(-0.17 * np.average(list_d_audio))
spit_concat = spit_audio * 0.5 + spit_landmark * 0.5

print(spit_landmark)
print(spit_audio)
print(spit_concat)

student_cap = cv2.VideoCapture(STUDENT_PATH)
teacher_cap = cv2.VideoCapture(TEACHER_PATH)

# Define font and other properties for the text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)  # White color for the text
thickness = 1
position1 = (50, 50)  # Position for the text in the first video
position2 = (50, 50)  # Position for the text in the second video
frame_count = 0
while (student_cap.isOpened()):
    success, image = student_cap.read()
    success_teach, image_teach = teacher_cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
    if not success_teach:
        # print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a selfie-view display.
    image = cv2.resize(image, (720, 360)) 
    image_teach = cv2.resize(image_teach, (720, 360)) 
    score_now1 = list_score_concat[frame_count]
    score_now2 = list_score_concat[frame_count]
    score_now3 = list_score_concat[frame_count]
    frame_count += 1
    score_now1 = math.floor(score_now1 * 100) / 100
    score_now2 = math.floor(score_now2 * 100) / 100
    score_now3 = math.floor(score_now3 * 100) / 100
    text_score1 = "Hand score: "+str(score_now1)
    text_score2 = "Audio score: "+str(score_now2)
    text_score3 = "Combine score: "+str(score_now3)
    # Get frame dimensions for bottom-right text placement
    frame1_height, frame1_width = image.shape[:2]
    # Calculate the size of the text to make sure it fits
    text_size, _ = cv2.getTextSize(text_score3, font, 0.75, thickness)

    # Set the position for the bottom-right corner of the top video
    text_x = frame1_width - text_size[0] - 15  # 10 pixels padding from the right
    text_y = frame1_height - 10  # 10 pixels padding from the bottom
    # Add text at the bottom-right corner of the top video (Video 1)
    cv2.putText(image, text_score1, (text_x, text_y), font, 0.75, color, thickness, cv2.LINE_AA)
    cv2.putText(image, text_score2, (text_x, text_y-text_size[1]), font, 0.75, color, thickness, cv2.LINE_AA)
    cv2.putText(image, text_score3, (text_x, text_y-(2*text_size[1])), font, 0.75, color, thickness, cv2.LINE_AA)
    cv2.putText(image, 'Student', position1, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(image_teach, 'Teacher', position2, font, font_scale, color, thickness, cv2.LINE_AA)


    canvas = cv2.vconcat([image, image_teach])

    cv2.imshow('MediaPipe Hands', canvas)
    
    # out_final.write(canvas)  

    if (cv2.waitKey(5) & 0xFF == 27):
        break

def Close(root):
    root.destroy()

root = tk.Tk()

landmark_score_label = tk.Label(root, text="Landmark Score:"+str(math.floor(spit_landmark * 100) / 100))

landmark_score_label.pack()

audio_score_label = tk.Label(root, text="Audio Score:"+str(math.floor(spit_audio * 100) / 100))

audio_score_label.pack()

combine_score_label = tk.Label(root, text="Combine Score:"+str(math.floor(spit_concat * 100) / 100))

combine_score_label.pack()

submit_button = tk.Button(root, text="Close", command= lambda: Close(root))
submit_button.pack(padx=20, pady=20)

root.mainloop()