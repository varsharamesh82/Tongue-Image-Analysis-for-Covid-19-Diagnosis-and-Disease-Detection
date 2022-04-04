#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:17:31 2020

@author: sanjana
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score

df=pd.read_csv('/Users/sanjana/Desktop/IP Paper/tonguedfff.csv')
df.head()

#df.drop([16, 18,19,21,22,25,27,28,29])
    

list(df.columns)
#df['Healthy/Disease'].value_counts()

df['sex'] = df['Gender'].map( {'Male':1, 'Female':0} )
df['color'] = df['colour'].map( {'pink':1, 'purple':2, 'red':3,'yellow':4,'white':5} )

X=df[['sex','area','central_height','central_width','height_width_ratio','color','smaller_half_dist',
 'circle_area',
 'circle_area_ratio',
 'num_contours',
 'area_contours',
 'len_contours']]

y = df['Covid'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
'''df['area']=sc.transform(df['area'])
df['height']=sc.transform(df['height'])
df['width']=sc.transform(df['width'])
df['height_width_ratio']=sc.transform(df['height_width_ratio'])'''

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


#SVM
from sklearn.svm import SVC 
svmclf = SVC() 
svmclf.fit(X_train, y_train) 
  
# Storing the predictions of the non-linear model 
y_pred = svmclf.predict(X_test) 
  

from sklearn.metrics import r2_score
# Evaluating the performance of the non-linear model 
print('Accuracy : '+str(accuracy_score(y_test, y_pred))) 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred=neigh.predict(X_test)
print('Accuracy : '+str(accuracy_score(y_test, y_pred))) 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

data = {'SVC':89, 'Logistic Regression':78, 'K-NN':89} 
courses = list(data.keys()) 
values = list(data.values()) 
   
fig = plt.figure(figsize = (6, 3)) 
  
# creating the bar plot 
plt.bar(courses, values, color=['blue', 'orange', 'green'] ,
        width = 0.3) 
  
plt.xlabel("Classifier") 
plt.ylabel("Accuracy (%)") 
plt.title("Heart Disease detection - Average Accuracy achieved in 100 executions ") 
plt.show() 
