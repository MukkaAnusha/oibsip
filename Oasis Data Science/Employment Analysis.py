#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


#Read the data from csv file
emp=pd.read_csv("Unemployment in India.csv")


# In[3]:


#Displaying the first 5 rows of the data
emp.head()


# In[4]:


#Displaying the bottom 5 rows of the data
emp.tail()


# In[5]:


emp.info()


# In[6]:


emp.isnull().sum()


# In[7]:


emp.columns=["States","Date","Frequency","Estimated Unemployment Rate(%)","Estimated Employed","Estimated Labour Participation Rate(%)","Region"]


# In[8]:


#Exclude non-numeric columns for correlation calculation
numeric_columns=["Estimated Unemployment Rate(%)","Estimated Employed","Estimated Labour Participation Rate(%)"]
numeric_data=emp[numeric_columns]


# In[9]:


plt.figure(figsize=(10,8))
sns.heatmap(numeric_data.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation matrix")
plt.show()


# In[10]:


#Histogram of Estimated Employed by Region
plt.figure(figsize=(12,10))
plt.title("Distribution of Estimated Employed Rate by Region")
sns.histplot(x="Estimated Employed",hue="Region",data=emp)
plt.xlabel("Estimated Employed")
plt.ylabel("Count")
plt.show()


# In[11]:


#Histogram of Estimated Unemployment Rate by Region
plt.figure(figsize=(12,10))
plt.title("Distribution of Estimated Unemployment Rate by Region")
sns.histplot(x="Estimated Unemployment Rate(%)",hue="Region",data=emp)
plt.xlabel("Estimated Unemployment Rate")
plt.ylabel("Count")
plt.show()


# In[12]:


aggregated_data=emp.groupby(["Region","States"],as_index=False)["Estimated Unemployment Rate(%)"].mean()
fig = px.sunburst(aggregated_data, path=["Region", "States"], values="Estimated Unemployment Rate(%)",
                  color_continuous_scale="RdYlGn", title="Interactive Pie Chart of Unemployment Rate in India")
fig.show()

