#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libraries
import pandas as pd
import math
import numpy as np 
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# In[2]:


# Importing the dataset 
data = pd.read_csv('train.csv') 


# In[3]:


data.head(5)


# In[4]:


# Splitting the dataset into Training set and Test set 
from sklearn.model_selection import train_test_split 
Y = data.fetal_health
X = data.drop('fetal_health',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 101)


# In[5]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
y_pred


# In[6]:


from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)    #using the trained model predict the output of the text_X input values
print("Accuracy: {:.0f}%".format(accuracy_score(Y_test , y_pred)*100))


# In[7]:


Test_data= pd.read_csv('test.csv') 
print('Test data',Test_data.shape)
Test_data_pred =model.predict(Test_data)

Test_data_pred


# In[8]:


output=[Test_data_pred]
df=pd.DataFrame(output)
output=df.to_csv(r'outcome.csv',index=None,header=True)
print(df)


# In[ ]:





# In[ ]:




