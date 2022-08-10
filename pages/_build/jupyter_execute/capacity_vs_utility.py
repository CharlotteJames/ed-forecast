#!/usr/bin/env python
# coding: utf-8

# # Separating Capacity and Utility 

# ## Overview 

# The capacity and utility features are highly correlated (see below). This can affect overall model performance and feature importances.
# 
# To understand the extent of this effect we fit separate models to capacity and utility features, and compare performance.
# 
# 
# - Capacity features: 111 calls offered, Ambulance answered, GP appointments available
# - Utility features: 111 calls answered, Ambulance made

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# ## Import data 

# In[3]:


dta = pd.read_csv('https://raw.githubusercontent.com/CharlotteJames/ed-forecast/main/data/master_scaled.csv',
                  index_col=0)


# In[4]:


dta.columns = ['_'.join([c.split('/')[0],c.split('/')[-1]]) 
               if '/' in c else c for c in dta.columns]


# In[5]:


dta.ccg.unique().shape


# ## Feature correlations 

# In[6]:


fig,ax_list = plt.subplots(1,2, figsize = (10,5))

xx = np.arange(0,140)*10

ax = ax_list[0]

ax.plot(dta['111_111_answered'].values, dta['111_111_offered'].values, 'ro')
ax.set_xlabel('111 Utility')
ax.set_ylabel('111 Capacity')
ax.plot(xx,xx, 'k--')

ax = ax_list[1]

ax.plot(dta['amb_sys_made'].values, dta['amb_sys_answered'].values, 'bo')
ax.set_xlabel('Ambulance Utility')
ax.set_ylabel('Ambulance Capacity')
ax.plot(xx,xx,'k--')

plt.show()


# ## Add random feature

# In[7]:


# Adding random features

rng = np.random.RandomState(0)
rand_var = rng.rand(dta.shape[0])
dta['rand1'] = rand_var


# In[8]:


dta.shape


# ## Fitting function 

# In[9]:


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


# ## Utility Model 

# In[10]:


features = ['111_111_answered', 'amb_sys_made', 'rand1']


# ### Linear Regression 

# In[11]:


model = LinearRegression()

results = fit_model(dta,model,features)


# #### Performance 

# In[12]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[13]:


coefs = pd.DataFrame(
   [model.coef_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ### Random Forest 

# In[14]:


model = RandomForestRegressor()

results = fit_model(dta,model,features)


# #### Performance 

# In[15]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[16]:


coefs = pd.DataFrame(
   [model.feature_importances_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ### Gradient Boosted Trees 

# In[17]:


model = GradientBoostingRegressor()

results = fit_model(dta,model,features)


# #### Performance 

# In[18]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[19]:


coefs = pd.DataFrame(
   [model.feature_importances_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ### Summary 

# Linear regression performs poorly, with an $R^2$ < 0.1
# 
# Random Forest overfits to the training data, and the random variable is comparatively important
# 
# Gradient boosted trees performs best and does not overfit. The random variable is ranked lowest.

# ## Capacity Model 

# In[20]:


features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered', 'rand1']


# ### Linear Regression 

# In[21]:


model = LinearRegression()

results = fit_model(dta,model,features)


# #### Performance 

# In[22]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[23]:


coefs = pd.DataFrame(
   [model.coef_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ### Random Forest 

# In[24]:


model = RandomForestRegressor()

results = fit_model(dta,model,features)


# #### Performance 

# In[25]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[26]:


coefs = pd.DataFrame(
   [model.feature_importances_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ### Gradient Boosted Trees 

# In[27]:


model = GradientBoostingRegressor()

results = fit_model(dta,model,features)


# #### Performance 

# In[28]:


res=pd.DataFrame()
res['test_score'] = results['test_score']
res['train_score'] = results['train_score']

res.describe()


# #### Coefficients 

# In[29]:


coefs = pd.DataFrame(
   [model.feature_importances_
    for model in results['estimator']],
   columns=features
)

coefs.describe()


# ### Summary 

# Linear regression again performs poorly.
# 
# Random Forest overfits to the training data, but the random variable is ranked lowest.
# 
# Gradient boosted trees performs best and does not overfit. The random variable is ranked lowest.
