import cv2
import pickle
import pandas as pd
import mediapipe as mp

# pickle_in = open("knnpickle_file","rb")
# example_dict = pickle.load(pickle_in)

data_path = "train.csv"
ds = pd.read_csv(data_path)

y = ds['label']
X = ds.drop('label', axis=1).to_numpy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=6)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

#Fit the model
knn.fit(X_train, y_train)

mp_pose = mp.solutions.pose
cap = cv2.VideoCapture('table_tennis_serve2.mp4')

all_list = []
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    while(True):

        ret, image = cap.read()
        results = pose.process(image)
        # image = cv2.flip(image, 1)

        order2 = [11, 12, 14, 16, 18, 20, 22, 23, 24]

        if results.pose_landmarks:
            data = []
            for j in order2:
                pose1 = results.pose_landmarks.landmark[j]
            
                data.append(pose1.x)
                data.append(pose1.y)
            # print(data)
            data = [data]
            new_output1 = knn.predict(data)
            pred_label = new_output1[0]
            # print(pred_label)

            new_output2 = knn.predict_proba(data)
            proba_label = new_output2[0][new_output1[0]]
            # print(proba_label)
            # all_list.append(data)

            cv2.putText(image, f'pred: {pred_label}', (40, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (248, 248, 255), 2, 4)
            cv2.putText(image, f'proba: {proba_label}', (40, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (248, 248, 255), 2, 4)

        else:
            print("No result.")
        
        # print(all_list)
        
        if ret:
            cv2.imshow('Video', image)
        else:
            break

        if cv2.waitKey(1) == ord('q'):
            break




















# import matplotlib
# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# import mediapipe as mp
# import cv2

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# cap = cv2.VideoCapture('table_tennis_serve1.mp4')

# data_path = "train.csv"
# ds = pd.read_csv(data_path)

# y = ds['label']
# X = ds.drop('label', axis=1)

# # i = len(X.columns)
# # print(i)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=6)

# #import KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier

# # for i,k in enumerate(neighbors):

# #Setup a knn classifier with k neighbors
# knn = KNeighborsClassifier(n_neighbors=3)


# all_list = []
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:

#     while True:
#         ret, image = cap.read()
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = pose.process(image)

#         h, w, _ = image.shape

#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         mp_drawing.draw_landmarks(
#             image,
#             results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(255,184,99), thickness=2, circle_radius=2))

#         order = [11, 12, 14, 16, 18, 20, 22, 23, 24]

#         if results.pose_landmarks:

#             data = []
#             for i in order:
#                 pose = results.pose_landmarks.landmark[i]
            
#                 data.append(pose.x)
#                 data.append(pose.y)
#             all_list.append(data)

#             print(all_list)
#             #Fit the model
#             knn.fit(X_train, y_train)
#             print(knn.predict(all_list))

#         else:
#             print("No result.")

        

#         # print(all_list)

        
#         # print(knn.predict_proba(X_test))

#         if ret:
#             cv2.imshow('Video', image)
#         else:
#             break

#         if cv2.waitKey(1) == ord('q'):
#             break