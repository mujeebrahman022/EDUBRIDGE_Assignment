#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
# Importing the dataset 
data = pd.read_csv('train.csv') 
Y = data.fetal_health
X = data.drop('fetal_health',axis=1)


# In[2]:


# Splitting the dataset into Training set and Test set 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, test_Y = train_test_split(X, Y, test_size = 0.2, random_state = 101)


# In[5]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()   #Create a Gaussian Classifier

gnb.fit(X_train, Y_train)   #Train the model using the training sets

y_pred = gnb.predict(X_test)   #Predict the response for test dataset
y_pred = gnb.predict(X_test)
y_pred


# In[6]:


#Import scikit-learn metrics module for accuracy_score
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))


# In[10]:


Validation_test = pd.read_csv('test.csv') 
val_pred =gnb.predict(Validation_test)

val_pred


# In[11]:


output=[val_pred]
df=pd.DataFrame(output)
new=df.to_csv(r'output.csv',index=None,header=True)
print(df)


# In[ ]:





# In[ ]:




