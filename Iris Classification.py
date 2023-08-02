#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required packages
#packages for data manipulation
import numpy as np
import pandas as pd
#packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


iris=pd.read_csv("Iris.csv")


# In[23]:


iris.shape


# In[4]:


#displaying top 5 rows of the dataset
iris.head()


# In[5]:


#displaying bottom 5 rows of the dataset
iris.tail()


# In[6]:


iris.info()


# In[7]:


iris.isnull().sum()


# In[8]:


X=iris.drop(['Id','Species'],axis=1)
y=iris['Species']


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[10]:


knn=KNeighborsClassifier(n_neighbors=3)
#fit the model
knn.fit(X_train,y_train)


# In[11]:


KNeighborsClassifier(n_neighbors=3)


# In[12]:


y_pred=knn.predict(X_test)


# In[13]:


accuracy=accuracy_score(y_test,y_pred)
print("Accuracy is:",accuracy)


# In[14]:


#Scatter plot
plt.scatter(iris["SepalLengthCm"],iris["SepalWidthCm"],color="y")
plt.xlabel("Sepal Length (in cm)")
plt.xlabel("Sepal Width (in cm)")
plt.show()


# In[15]:


#joint plot
sns.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=iris,height=5)
plt.show()


# In[16]:


sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm",data=iris,hue="Species",palette="rocket")
plt.show()


# In[17]:


#displot
sns.displot(iris.PetalWidthCm,bins=10,color="violet",rug="True",kde="True")
plt.show()


# In[18]:


#Box plot
sns.boxplot(x="Species",y="PetalLengthCm",data=iris)
plt.show()


# In[19]:


#violin plot
sns.violinplot(x="Species",y="PetalLengthCm",data=iris,height=6)
plt.show()


# In[21]:


#pair plot
sns.pairplot(iris.drop("Id",axis=1),hue="Species",height=3,palette="Set2")
plt.show()


# In[22]:


iris.drop("Id",axis=1).boxplot(by="Species",figsize=(12,6))
plt.show()

