#!/usr/bin/env python
# coding: utf-8

# # Tree-based models 

# ## Overview 

# This notebook contains an initial exploration of tree-based regressions to predict monthly ED demand. 
# 
# As the variables population, people, places and lives only vary annually they cannot be included in the model due to data leakage between the training and test sets.
# 
# For all models, variables used include:
# 
# - Service capacity (111, GP, Ambulance)
# - Service utility (111, Ambulance)

# In[1]:


#turn warnings off to keep notebook tidy
import warnings
warnings.filterwarnings('ignore')


# ## Import libraries 

# In[2]:


import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold


# ## Import data 

# In[3]:


dta = pd.read_csv('https://raw.githubusercontent.com/CharlotteJames/ed-forecast/main/data/master_scaled_new.csv',
                  index_col=0)


# In[4]:


dta.columns = ['_'.join([c.split('/')[0],c.split('/')[-1]]) 
               if '/' in c else c for c in dta.columns]


# In[5]:


dta.ccg.unique().shape


# ## Add random feature

# In[6]:


# Adding random features

rng = np.random.RandomState(0)
rand_var = rng.rand(dta.shape[0])
dta['rand1'] = rand_var


# In[7]:


dta.shape


# ## Fitting function 

# In[8]:


def fit_model(dta, model, features):
    
    
    y = dta['ae_attendances_attendances']
    X = dta[features]
    
    #cross validate to get errors on performance and coefficients
    cv_model = cross_validate(model, X,y, 
                            cv=RepeatedKFold(n_splits=5, n_repeats=5,
                                             random_state=0),
                            return_estimator=True, 
                              return_train_score=True, n_jobs=2)
    
    clf = model.fit(X, y)

    
    return cv_model


# ## Model Comparison

# ### Random Forest 

# In[9]:


model = RandomForestRegressor()

features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered',
            '111_111_answered', 'amb_sys_made']

results = fit_model(dta,model,features)


# #### Performance 

# In[10]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[11]:


coefs = pd.DataFrame(
   [model.feature_importances_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ### Extra Trees

# In[12]:


model = ExtraTreesRegressor()

features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered',
            '111_111_answered', 'amb_sys_made']

results = fit_model(dta,model,features)


# #### Performance 

# In[13]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[14]:


coefs = pd.DataFrame(
   [model.feature_importances_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ### Gradient Boosted Trees

# In[15]:


model = GradientBoostingRegressor()

features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered',
            '111_111_answered', 'amb_sys_made']

results = fit_model(dta,model,features)


# #### Performance 

# In[16]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[17]:


coefs = pd.DataFrame(
   [model.feature_importances_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ### ADA Boost

# In[18]:


model = AdaBoostRegressor()

features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered',
            '111_111_answered', 'amb_sys_made']

results = fit_model(dta,model,features)


# #### Performance 

# In[19]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[20]:


coefs = pd.DataFrame(
   [model.feature_importances_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ## Summary 

# - Extra Trees does not preform well
# - Random forest with default parameters is overfitting to the training data
# - Gradient boosted trees performs best
