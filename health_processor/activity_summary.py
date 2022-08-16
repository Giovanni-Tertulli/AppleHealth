#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sysconfig import get_python_version
import pandas as pd
import matplotlib.pyplot as plt

get_python_version().run_line_magic('matplotlib', 'inline')


# In[12]:


df = pd.read_csv("/Users/giovanni/Desktop/portfolio_projects/p2/qs_ledger-master/apple_health/data/ActivitySummary.csv")
df


# In[13]:


df.dtypes


# In[15]:


df.index = pd.to_datetime(df['dateComponents'], format="%Y-%m-%d")
(df.groupby(by=[df.index.year, df.index.month])["activeEnergyBurned"].mean()).hist()


# In[16]:


print("Your data has been processed!")


# In[ ]:




