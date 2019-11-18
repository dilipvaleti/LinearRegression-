#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)

#Reading data
data=pd.read_csv('headbrain.csv')
print(data.shape)
data.head()


# In[56]:


#collection x and y as arrays
x=data['Head Size(cm^3)'].values
y=data['Brain Weight(grams)'].values


# In[51]:


#mean x and y
mean_x=np.mean(x)
mean_y=np.mean(y)

#total number of values
n=len(x)

#using formula to calculate b1 and b0
numer=0
denom=0
for i in range(n):
    numer += (x[i]-mean_x)*(y[i]-mean_y)
    denom += (x[i]-mean_x)**2
b1=numer/denom
b0= mean_y - (b1*mean_x)
#Print coefficients
print(b1,b0)


# In[61]:


#plotting values and regresiion line

max_x=np.max(x)+100
min_x=np.min(x)-100

#calculating line values x and y
x_new=np.linspace(min_x, max_x, 1000)
y_new=b0+b1*x_new

#plotting line
plt.plot(x_new,y_new,color='#58b970',label='Regression line')

#plot scatter plot
plt.scatter(x,y,color='#ef5432',label='Scatterplot')
plt.xlabel('Head size in cm3')
plt.ylabel('Brain size in grms')
plt.legend()
plt.show()


# In[70]:


ss_t=0
ss_r=0
for i in range(n):
    y_pred=b0+b1*x[i]
    ss_t += (y[i]-mean_y)**2
    ss_r += (y_pred-mean_y)**2
r2=(ss_r/ss_t)
print(r2)


# #  Regression ussing Scikit learn

# In[75]:



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#cannot use rank 1 matrix in scikit learn
x=x.reshape((n,1))
#creating model
reg=LinearRegression()
#fitting training data
reg=reg.fit(x,y)
# y prediction
y_pred=reg.predict(x)
#calculatinng R2 score
r2_score=reg.score(x,y)
print(r2_score)
    
    


# In[ ]:




