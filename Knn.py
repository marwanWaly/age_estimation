from google.colab import drive
drive.mount('/content/gdrive')


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import cv2
import os


def image_to_feature_vector(image, size=(50, 50)):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
  return cv2.resize(image, size).flatten()



path_c=os.listdir('/content/UTKFace')


data = []
labels= []



for  imagePath in path_c:
    image = cv2.imread('/content/UTKFace/'+imagePath)
   
  
    label = imagePath.split(os.path.sep)[-1].split("_")[0]
    
    main_age=int(label)
    
    pixels = image_to_feature_vector(image)
    data.append(pixels)
 
    
    
    if  1 <= main_age < 5:
      labels.append(0)
    elif 5 < main_age <= 10:
      labels.append(1) 
    elif 10 < main_age <= 15:
      labels.append(2) 
    elif 15 < main_age <= 20:
      labels.append(3) 
    elif 20 < main_age <= 30:
      labels.append(4) 
    elif 30 < main_age <= 40:
      labels.append(5) 
    elif 40 < main_age <= 50:
      labels.append(6) 
    elif 50 < main_age <= 60:
      labels.append(7) 
    elif 60 < main_age <= 70:
      labels.append(8) 
    elif 70 < main_age <= 80:
      labels.append(9) 
    else :
      labels.append(10)
      



data=np.array(data)
labels=np.array(labels)


knn_cv = KNeighborsClassifier(n_neighbors=5)
cv_scores_v = cross_val_score(knn_cv, data, labels, cv=5)


print("feature vector with cross vlidation accuracy:")
print(np.mean(cv_scores_v))







(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
knn.predict(X_test)
prediction = knn.predict(X_test)
accuracy_v=metrics.accuracy_score(Y_test,prediction)


print("feature vector without cross vlidation accuracy:")
print(accuracy_v)


