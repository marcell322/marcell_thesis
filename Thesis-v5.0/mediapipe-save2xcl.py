import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from_image = False

if from_image:
    IMAGE_FILES = [
        "Rosseau hand.png"
    ]
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)

            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            flip = cv2.flip(annotated_image, 1)
            cv2.imwrite(
                'tmp/' + str(idx) + '.png', flip)
            plt.imshow(flip)
            plt.show()
            # Draw hand world landmarks.
            if not results.multi_hand_world_landmarks:
                continue
            # for hand_world_landmarks in results.multi_hand_world_landmarks:
            #     mp_drawing.plot_landmarks(
            #         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
# For webcam input:
else:
    # cap = cv2.VideoCapture('Faded_pianella.mp4')
    cap = cv2.VideoCapture('Faded.mp4')
    wi = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fr = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    out = cv2.VideoWriter('output_roseau.mp4',fourcc, fr, (wi, hi))  
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
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
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

                    # for i in range(len(pose_tangan)):
                    #     hand_landmarks.landmark[i].x = hand_landmarks.landmark[i].x * image.shape[0]
                    #     hand_landmarks.landmark[i].y = hand_landmarks.landmark[i].y * image.shape[1]
                    #     data_tangan.update(
                    #         {
                    #             pose_tangan[i] : f'{frame_ctr}'+", " +f'{hand_landmarks.landmark[i].x}' +", " +f'{hand_landmarks.landmark[i].y}'
                    #         }
                    #     )
                    # alldata.append(data_tangan)
                    # no_frame.append(frame_ctr)
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
    # df = pd.DataFrame(alldata)
    # df.to_excel("koordinat_faded_pianella.xlsx")
    # df.to_excel("koordinat_faded_roseau.xlsx")
    cap.release()
    out.release()  
    cv2.destroyAllWindows()  