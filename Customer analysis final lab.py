#!/usr/bin/env python
# coding: utf-8

# # Lab | Customer Analysis Final Round
# For this lab, we still keep using the marketing_customer_analysis.csv file that you can find in the files_for_lab folder.
# 
# It's time to put it all together. Remember the previous rounds and follow the steps as shown in previous lectures.
# 
# 01 - Problem (case study)
# Data Description.
# Goal.
# 02 - Getting Data
# Read the .csv file.
# 03 - Cleaning/Wrangling/EDA
# Change headers names.
# Deal with NaN values.
# Categorical Features.
# Numerical Features.
# Exploration.
# 04 - Processing Data
# Dealing with outliers.
# Normalization.
# Encoding Categorical Data.
# Splitting into train set and test set.
# 05 - Modeling
# Apply model.
# 06 - Model Validation
# R2.
# MSE.
# RMSE.
# MAE.
# 07 - Reporting
# Present results.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os #we will use the function listdir to list files in a folder
import math #to apply absolute value
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')


# In[2]:


dataframe = pd.read_csv("marketing_customer_analysis (1).csv")
dataframe


# In[3]:


dataframe.isnull()


# In[4]:


num = dataframe.select_dtypes(include='number')
num


# In[5]:


cat = dataframe.select_dtypes(exclude='number')
cat


# In[6]:


df = num.dropna()
df


# In[7]:


df['Total Claim Amount']


# In[8]:


sns.boxplot(x = df['Total Claim Amount'])
plt.show()


# In[9]:


sns.distplot(x = df['Total Claim Amount'])
plt.show()


# In[10]:


df[['Total Claim Amount']].describe()


# In[16]:


X = df["Income"]
y = df['Total Claim Amount']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=0)
r_squared_value = r2_score(X, y)
print(r_squared_value)
R = r_squared_value**2
print(R)


# In[12]:


print('X_train : ')
print(X_train.head())
print('')
print('X_test : ')
print(X_test.head())
print('')
print('y_train : ')
print(y_train.head())
print('')
print('y_test : ')
print(y_test.head())


# In[ ]:





# In[ ]:




