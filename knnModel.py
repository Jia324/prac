import matplotlib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pickle


data_path = "train.csv"
ds = pd.read_csv(data_path)

y = ds['label']
X = ds.drop('label', axis=1).to_numpy()

# i = len(X.columns)
# print(i)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=6)

#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# for i,k in enumerate(neighbors):

#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=3)

#Fit the model
knn.fit(X_train, y_train)

f = open('knnpickle_file', 'wb') 
# source, destination 
pickle.dump(knn, f)   

# print(knn.score(X_test, y_test))
# print(X_test.shape)
# print(knn.predict(X_test))
# print(knn.predict_proba(X_test))

new_input = [[0.554373801,0.430265397,0.486411512,0.423677564,0.464797407,0.494717032,0.480314106,0.445444643,0.479922354,0.429117829,0.480599225,0.415670663,0.482960224,0.42295754,0.545605004,0.554132462,0.512951195,0.558003485]]
# new_input = [[0.580253005,0.414981365,0.519001961,0.41924122,0.499236494,0.489822388,0.518487394,0.421522677,0.521344066,0.402132213,0.523593903,0.391154706,0.524600267,0.399746358,0.570460618,0.538744986,0.540237248,0.54212445]]


# get prediction for new input
new_output1 = knn.predict(new_input)
print(new_output1)
new_output2 = knn.predict_proba(new_input)
print(new_output2[0][new_output1[0]])

    







# #Setup arrays to store training and test accuracies
# neighbors = np.arange(1,9)
# train_accuracy =np.empty(len(neighbors))
# test_accuracy = np.empty(len(neighbors))

# #Compute accuracy on the training set
# train_accuracy[i] = knn.score(X_train, y_train)

# #Compute accuracy on the test set
# test_accuracy[i] = knn.score(X_test, y_test)

# #Generate plot
# plt.title('k-NN Varying number of neighbors')
# plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
# plt.plot(neighbors, train_accuracy, label='Training accuracy')
# plt.legend(loc='upper right')
# plt.xlabel('Number of neighbors')
# plt.ylabel('Accuracy')
# plt.show()
