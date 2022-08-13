import cv2
import mediapipe as mp
import csv
import os
from glob import glob

dataset_path = 'DATASET'

print(f"subfolders = {os.listdir(dataset_path)}")


mp_pose = mp.solutions.pose
all_list = []

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    order1 = ['label', '11_x', '11_y', '12_x', '12_y', '14_x', '14_y', '16_x', '16_y', '18_x', '18_y', '20_x', '20_y', '22_x', '22_y', '23_x', '23_y', '24_x', '24_y']
    data1 = []
    for n in range(19):
        data1.append(order1[n])
    all_list.append(data1)

    # for i in range(5):
    i = 0
    for subfolder in os.listdir(dataset_path):
        
        label = os.listdir(dataset_path)

        path = os.path.join(dataset_path, subfolder)
        path = glob(os.path.join(path, '**'))
        print(path)

    #     for img in path:
    #         image = cv2.imread(img)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #         results = pose.process(image)
            
    #         order2 = [11, 12, 14, 16, 18, 20, 22, 23, 24]

    #         if results.pose_landmarks:
    #             data = []
    #             data.append(os.listdir(dataset_path).index(label[i]))
    #             for j in order2:
    #                 pose1 = results.pose_landmarks.landmark[j]
                
    #                 data.append(pose1.x)
    #                 data.append(pose1.y)
    #             all_list.append(data)
    #         else:
    #             print("No result.")
    #     i += 1

    #     # print(all_list)
                

    # with open('train.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerows(all_list)

                
