#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[74]:


cd C:\Users\ss\Desktop\Data science material\Linear Regression


# In[75]:


companies=pd.read_csv('Startups.csv')
x=companies.iloc[:,:-1].values
y=companies.iloc[:,-1].values
companies.head()


# In[76]:


#Data visualization
#Build the correlation matrix
sns.heatmap(companies.corr())


# In[77]:


#Emcoading categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])

onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]


# In[78]:


#Splitting the data into the training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[79]:


#Fitting multiple linear regresssion to the training set

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)


# In[80]:


#Predict the test results
y_pred=regression.predict(X_test)


# In[81]:


#calculate the coeffecients
print(regression.coef_)


# In[82]:


#calculate the intercept
print(regression.intercept_)


# In[83]:


#calculate the R sqaure value
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

# R 2 is 0.9347 is a good value and this is a good model


# In[ ]:




