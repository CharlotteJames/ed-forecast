#!/usr/bin/env python
# coding: utf-8

# # KNN imputation 

# In[1]:


import os
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer


# In[2]:


master = pd.read_csv('https://raw.githubusercontent.com/CharlotteJames/ed-forecast/main/data/master_scaled_impute.csv',
                  index_col=0)


# In[3]:


master.head()


# In[4]:


master.shape


# In[5]:


cols = master.columns[2:]


# ## KNN

# In[6]:


get_ipython().run_cell_magic('time', '', "\nfor N in range(3,12):\n    \n    knn = KNNImputer(n_neighbors = N)\n\n    master_imputed = knn.fit_transform(master[cols])\n    \n    master_imputed = pd.DataFrame(np.array(master_imputed), columns = cols )\n    master_imputed['month'] = master['month']\n    master_imputed['ccg'] = master['ccg']\n    \n    master_imputed.to_csv(f'../data/imputed/master_imputed_{N}.csv')")


# In[ ]:




