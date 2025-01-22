import cv2
from pathlib import Path
from moviepy import *
import librosa
import librosa.display
import numpy as np
import pandas as pd
from dtaidistance import dtw,dtw_ndim
import math

WINDOWING_SIZE = 5
WINDOW_SIZE = 1
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
    
    frame_cqt_indices = np.array(frame_cqt_indices, dtype=float)

    return frame_cqt_indices

STUDENT_NAME= "D:\Thesis Revision\marcell_thesis\media\Fur Elise - Lettre.mp4"
TEACHER_NAME= "D:\Thesis Revision\marcell_thesis\media\Fur Elise - Paul Barton.mp4"

# start detecting student
student_cap = cv2.VideoCapture(STUDENT_NAME)
student_wi = int(student_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
student_hi = int(student_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
student_fr = int(student_cap.get(cv2.CAP_PROP_FPS))
student_frame = int(student_cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
student_file = Path(STUDENT_NAME).stem
STUDENT_PATH = fr'..\media\mp_'+student_file+'.mp4'

# Audio Student to excel
student_audio_file = fr'..\media\audio_'+student_file+'_audio.wav'

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

# Loading student coordinate & angle
student_coordinate = []
student_angle = []
student_frame = []

print("opening student file")
student_excel = pd.read_excel(fr'..\media\coordinate_'+student_file+'.xlsx')
for i in (range(len(student_excel['WRIST']))):
    student_coordinate.append([])
    for pose in (HAND_POSE):
        temp = (student_excel[pose][i].split(', '))
        student_coordinate[i].append([eval(temp[1]),eval(temp[2])])
    student_angle.append(finger_angle_2joints(student_coordinate,i,TWO_JOINTS))
    student_frame.append(temp[0])

student_angle = np.array(student_angle)


# start detecting teacher
teacher_cap = cv2.VideoCapture(TEACHER_NAME)
teacher_wi = int(teacher_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
teacher_hi = int(teacher_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
teacher_fr = int(teacher_cap.get(cv2.CAP_PROP_FPS))
teacher_frame = int(teacher_cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
teacher_file = Path(TEACHER_NAME).stem
TEACHER_PATH = fr'..\media\mp_'+teacher_file+'.mp4'

# Audio Teacher to excel
teacher_audio_file = fr'..\media\audio_'+teacher_file+'_audio.wav'

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

# Loading teacher coordinate & angle
teacher_coordinate = []
teacher_angle = []

print("opening teacher file")
teacher_excel = pd.read_excel(fr'..\media\coordinate_'+teacher_file+'.xlsx')
for i in (range(len(teacher_excel['WRIST']))):
    teacher_coordinate.append([])
    for pose in (HAND_POSE):
        temp = (teacher_excel[pose][i].split(', '))
        teacher_coordinate[i].append([eval(temp[1]),eval(temp[2])])
    teacher_angle.append(finger_angle_2joints(teacher_coordinate,i,TWO_JOINTS))

teacher_angle = np.array(teacher_angle)

list_d_landmark = []
list_d_audio = []
list_d_concat = []

list_score_landmark = []
list_score_audio = []
list_score_concat = []

student_angle_normalized = (student_angle - student_angle.min()) / (student_angle.max() - student_angle.min())
teacher_angle_normalized = (teacher_angle - teacher_angle.min()) / (teacher_angle.max() - teacher_angle.min())
for i in range(len(student_cqt_to_frames)):
    j = ((i - 1) // WINDOWING_SIZE) * WINDOWING_SIZE
    windowed_student_landmarks = student_angle_normalized[j:j + WINDOWING_SIZE]
    windowed_teacher_landmarks = teacher_angle_normalized[j:j + WINDOWING_SIZE]
    windowed_student_cqt = student_cqt_to_frames[j:j + WINDOWING_SIZE]
    windowed_teacher_cqt = teacher_cqt_to_frames[j:j + WINDOWING_SIZE]
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
thickness = 2
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
    score_now = list_score_landmark[frame_count]
    frame_count += 1
    score_now = math.floor(score_now * 100) / 100
    text_score = "Score: "+str(score_now)
    # Get frame dimensions for bottom-right text placement
    frame1_height, frame1_width = image.shape[:2]
    # Calculate the size of the text to make sure it fits
    text_size, _ = cv2.getTextSize(text_score, font, font_scale, thickness)

    # Set the position for the bottom-right corner of the top video
    text_x = frame1_width - text_size[0] - 10  # 10 pixels padding from the right
    text_y = frame1_height - 10  # 10 pixels padding from the bottom
    # Add text at the bottom-right corner of the top video (Video 1)
    cv2.putText(image, text_score, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, 'Student', position1, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(image_teach, 'Teacher', position2, font, font_scale, color, thickness, cv2.LINE_AA)


    canvas = cv2.vconcat([image, image_teach])

    cv2.imshow('MediaPipe Hands', canvas)
    
    # out_final.write(canvas)  

    if (cv2.waitKey(5) & 0xFF == 27):
        break