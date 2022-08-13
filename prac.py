import cv2
import numpy as np
import mediapipe as mp
import os

pic_path_1 = 'DATASET/pic_1'
pic_path_2 = 'DATASET/pic_2'
pic_path_3 = 'DATASET/pic_3'


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


cap = cv2.VideoCapture('table_tennis_serve1.mp4')

a, b, c, r, point= 1, 1, 1, 20, 16
w1, h1 = 600, 400
w2, h2 = 570, 320 
w3, h3 = 680, 290

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while True:
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        h, w, _ = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,184,99), thickness=2, circle_radius=2))

        # image = cv2.flip(image, 1)
        
        if results.pose_landmarks:

            
            poseX = int(results.pose_landmarks.landmark[point].x * w)
            poseY = int(results.pose_landmarks.landmark[point].y * h)

            
            print(f"X: {poseX}, Y: {poseY}")

            if w1-r < poseX < w1+r and h1-r < poseY < h1+r:

                image_Name_1 = os.path.join(pic_path_1 , f'photo-{a}.jpg')
                cv2.imwrite(image_Name_1 , image)
                print(f'a:{a}')    
                a += 1

            if w2-r < poseX < w2+r and h2-r < poseY < h2+r:

                image_Name_2 = os.path.join(pic_path_2 , f'photo-{b}.jpg')
                cv2.imwrite(image_Name_2 , image)
                print(f'b:{b}')    
                b += 1

            if w3-r < poseX < w3+r and h3-r < poseY < h3+r:

                image_Name_3 = os.path.join(pic_path_3 , f'photo-{c}.jpg')
                cv2.imwrite(image_Name_3 , image)
                print(f'c:{c}')    
                c += 1
            
             

        
        cv2.circle(image, (w1, h1), r, (220, 248, 255), cv2.FILLED)
        cv2.circle(image, (w2, h2), r, (220, 248, 255), cv2.FILLED)
        cv2.circle(image, (w3, h3), r, (220, 248, 255), cv2.FILLED)

        if ret:
            cv2.imshow('Video', image)
        else:
            break

        if cv2.waitKey(1) == ord('q'):
            break

        # cv2.imwrite('image.jpg', image)