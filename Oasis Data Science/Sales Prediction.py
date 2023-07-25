#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing the packages required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Reading the csv file
data=pd.read_csv("Advertising.csv")


# In[4]:


data


# In[5]:


data.shape


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.columns


# In[11]:


data.duplicated().sum()


# In[12]:


data.isnull().sum()


# In[13]:


plt.figure(figsize=(4,4))
sns.scatterplot(data=data,x=data['TV'],y=data['Sales'])
plt.show()


# In[14]:


plt.figure(figsize=(4,4))
sns.scatterplot(data=data,x=data['Radio'],y=data['Sales'])
plt.show()


# In[15]:


plt.figure(figsize=(4,4))
sns.scatterplot(data=data,x=data['Newspaper'],y=data['Sales'])
plt.show()


# In[16]:


X=data.drop('Sales',axis=1)


# In[17]:


y=data['Sales']
y


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=0)


# In[19]:


from sklearn.linear_model import LinearRegression
sale=LinearRegression()


# In[20]:


sale.fit(X_train,y_train)


# In[21]:


prediction=sale.predict(X_test)


# In[22]:


prediction


# In[23]:


from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(prediction,y_test))
print('RMSE:',np.sqrt(metrics.mean_squared_error(prediction,y_test)))
print('R-Squared',metrics.r2_score(prediction,y_test))


# In[ ]:




