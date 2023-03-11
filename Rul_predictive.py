#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import pickle


# In[2]:


#Reading the data
columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
         "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
         ,"sensor20","sensor21"]
df_train_FD001 = pd.read_csv("https://raw.githubusercontent.com/Chandrashekhar0106/ml_predictive_maintenance_Jet_engine/main/train_FD001.txt",sep='\s+',names=columns)
df_train_FD002 = pd.read_csv("https://raw.githubusercontent.com/Chandrashekhar0106/ml_predictive_maintenance_Jet_engine/main/train_FD002.txt",sep='\s+',names=columns)
df_train_FD003 = pd.read_csv("https://raw.githubusercontent.com/Chandrashekhar0106/ml_predictive_maintenance_Jet_engine/main/train_FD003.txt",sep='\s+',names=columns)
df_train_FD004 = pd.read_csv("https://raw.githubusercontent.com/Chandrashekhar0106/ml_predictive_maintenance_Jet_engine/main/train_FD004.txt",sep='\s+',names=columns)
df_test_FD001 = pd.read_csv("https://raw.githubusercontent.com/Chandrashekhar0106/ml_predictive_maintenance_Jet_engine/main/test_FD001.txt",sep='\s+',names=columns)
df_test_FD002 = pd.read_csv("https://raw.githubusercontent.com/Chandrashekhar0106/ml_predictive_maintenance_Jet_engine/main/test_FD002.txt",sep='\s+',names=columns)
df_test_FD003 = pd.read_csv("https://raw.githubusercontent.com/Chandrashekhar0106/ml_predictive_maintenance_Jet_engine/main/test_FD003.txt",sep='\s+',names=columns)
df_test_FD004 = pd.read_csv("https://raw.githubusercontent.com/Chandrashekhar0106/ml_predictive_maintenance_Jet_engine/main/test_FD004.txt",sep='\s+',names=columns)


# In[3]:


df_train_FD001.head()


# In[4]:


#Adding RUL column to each dataset
def add_RUL_column(df):
    train_grouped_by_unit = df.groupby('id') 
    max_time_cycles = train_grouped_by_unit['cycle'].max() 
    merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='id',right_index=True)
    merged["RUL"] = merged["max_time_cycle"] - merged['cycle']
    merged = merged.drop("max_time_cycle", axis=1) 
    return merged


# In[5]:


df_train_FD001 = add_RUL_column(df_train_FD001)
df_train_FD001.head()


# In[6]:


x= df_train_FD001.drop(columns='RUL')


# In[7]:


x


# In[8]:


y = df_train_FD001['RUL']


# In[9]:


y


# In[ ]:




