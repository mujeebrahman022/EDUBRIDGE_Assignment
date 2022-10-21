#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
# Importing the dataset 
data = pd.read_csv('train.csv') 


# In[4]:


# Splitting the dataset into Training set and Test set 
from sklearn.model_selection import train_test_split 
Y = data.fetal_health
X = data.drop('fetal_health',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 101)


# In[5]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, Y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=7, p=2,
                     weights='uniform')


# In[6]:


print('X_train', X_train.shape)
print('y_train', Y_train.shape)

print('X_test', X_test.shape)
print('y_test', Y_test.shape)


# In[7]:


#Predict the response for test dataset
y_pred =model.predict(X_test)

y_pred


# In[8]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score
# Model Accuracy, how often is the classifier correct?
print("Accuracy: {:.0f}".format(accuracy_score(Y_test, y_pred)*100))


# In[9]:


Validation_test = pd.read_csv('test.csv') 
print('Validation_test',Validation_test.shape)


# In[10]:


val_pred =model.predict(Validation_test)

val_pred


# In[11]:


output=[val_pred]
df=pd.DataFrame(output)
new=df.to_csv(r'output.csv',index=None,header=True)
print(df)


# In[ ]:




