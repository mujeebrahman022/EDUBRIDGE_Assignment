#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
#read data into dataframe
data = pd.read_csv ('train.csv')


# In[5]:


#Examine the first few rows 
print(data.head())


# In[6]:



#split data into train and test sets
Y = data.fetal_health
X = data.drop('fetal_health',axis=1)
train_X, test_X, train_Y, test_Y = train_test_split(X,Y, test_size = 0.2)


# In[7]:


print('train_X', train_X.shape)
print('train_Y', train_Y.shape)
print('test_X', test_X.shape)
print('test_Y', test_Y.shape)


# In[8]:


#decide on the model
model = DecisionTreeClassifier()
# fit the model to the training set
model.fit(train_X, train_Y)


# In[21]:


from sklearn.metrics import accuracy_score
#using the trained model predict the output of the text_X input values
y_pred = model.predict(test_X)
# Model Accuracy, how often is the classifier correct?
print("Accuracy: {:.0f}%".format(accuracy_score(test_Y, y_pred)*100))


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





# In[ ]:




