#!/usr/bin/env python
# coding: utf-8

# # Developing the Capacity Model

# ## Overview 

# This notebook contains the code to develop the capacity model. 
# 
# Initially 3 different models are compared (Linear regression, Random Forest regresstion, Gradient Boosted regression).
# 
# Hyper-parameters of the best model are fine-tunes to maximise performance in unseen data while preventing over-fitting and minimising model complexity

# In[1]:


#turn warnings off to keep notebook tidy
import warnings
warnings.filterwarnings('ignore')


# ## Import libraries 

# In[2]:


import os
import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold


import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# ## Import data 

# In[3]:


dta = pd.read_csv('../data/master_scaled.csv', index_col=0)


# In[4]:


dta.columns = ['_'.join([c.split('/')[0],c.split('/')[-1]]) 
               if '/' in c else c for c in dta.columns]


# In[5]:


dta.head()


# In[6]:


dta.shape


# ### Add random feature

# In[7]:


# Adding random features

rng = np.random.RandomState(0)
rand_var = rng.rand(dta.shape[0])
dta['rand1'] = rand_var


# In[8]:


dta.shape


# ## Model Comparison

# Features in the dataset that measure service capacity are:
# 
# - *gp_appt_available*: the number of GP appointments available per 10,000 people per month
# - *111_111_offered*: the number of 111 calls offered (i.e. that the service can answer) per 10,000 people per month
# - *amb_sys_answered*: the number of calls answered by the ambulance system per 10,000 people per month

# ### Pair plot 

# In[9]:


fig = sns.pairplot(dta[['gp_appt_available',
                        '111_111_offered', 'amb_sys_answered', 'rand1',
                        'ae_attendances_attendances']]\
                   .select_dtypes(include=np.number), 
                   kind="reg",
                   plot_kws={'line_kws':{'color':'black'}, 
                            'scatter_kws': 
                             {'color':'green','alpha': 0.1}},
                   diag_kws={'color':'blue'})

#plt.savefig('capacity_pair.png')


# ### Linear regression 

# In[10]:


model = LinearRegression()

features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered', 'rand1']

    
y = dta['ae_attendances_attendances']
X = dta[features]

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
    
scores_train, scores_test, feats = [],[],[]
    
for train_index, test_index in cv.split(X, y):
        
    model.fit(X.iloc[train_index], y.iloc[train_index])
        
    scores_test.append(model.score(X.iloc[test_index],
                                   y.iloc[test_index]))
    scores_train.append(model.score(X.iloc[train_index],
                                    y.iloc[train_index]))

    feats.append(model.coef_)


# #### Performance 

# In[11]:


results=pd.DataFrame()
results['train'] = scores_train
results['test'] = scores_test


# In[12]:


results.describe()


# #### Feature Importance 

# In[13]:


feat_imp = pd.DataFrame()

for i,f in enumerate(features):
    
    feat_imp[f] = np.array(feats)[:,i]

feat_imp.describe()


# ### Random forest 

# In[14]:


model = RandomForestRegressor(max_depth=5, n_estimators=5,
                              random_state=0)
#model = GradientBoostingRegressor(max_depth=5, n_estimators=5)

features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered', 'rand1']

    
y = dta['ae_attendances_attendances']
X = dta[features]

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
    
scores_train, scores_test, feats = [],[],[]
    
for train_index, test_index in cv.split(X, y):
        
    model.fit(X.iloc[train_index], y.iloc[train_index])
        
    scores_test.append(model.score(X.iloc[test_index],
                                   y.iloc[test_index]))
    scores_train.append(model.score(X.iloc[train_index],
                                    y.iloc[train_index]))

    feats.append(model.feature_importances_)


# #### Performance 

# In[15]:


results=pd.DataFrame()
results['train'] = scores_train
results['test'] = scores_test


# In[16]:


results.describe()


# #### Feature importance 

# In[17]:


feat_imp = pd.DataFrame()

for i,f in enumerate(features):
    
    feat_imp[f] = np.array(feats)[:,i]

feat_imp.describe()


# ### Gradient boosted tress

# In[18]:


model = GradientBoostingRegressor(max_depth=5, n_estimators=5, 
                                  random_state=1)

features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered', 'rand1']

    
y = dta['ae_attendances_attendances']
X = dta[features]

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
    
scores_train, scores_test, feats = [],[],[]
    
for train_index, test_index in cv.split(X, y):
        
    model.fit(X.iloc[train_index], y.iloc[train_index])
        
    scores_test.append(model.score(X.iloc[test_index],
                                   y.iloc[test_index]))
    scores_train.append(model.score(X.iloc[train_index],
                                    y.iloc[train_index]))
    
    feats.append(model.feature_importances_)


# #### Performance 

# In[19]:


results=pd.DataFrame()
results['train'] = scores_train
results['test'] = scores_test


# In[20]:


results.describe()


# #### Feature Importance 

# In[21]:


feat_imp = pd.DataFrame()

for i,f in enumerate(features):
    
    feat_imp[f] = np.array(feats)[:,i]

feat_imp.describe()


# ### Summary 

# **Linear Regression**
# 
# - Very poor performance, mean $R^2$ ~ 0.1
# 
# **Random Forest**
# 
# - Best performance with mean $R^2$ = 0.4 in test data
# - Feature importance is stable: ambulance capacity is most important, followed by 111 then GP capacity. 
# - The random feature has low importnace
# 
# **Gradient Boosted Trees**
# 
# - Doesn't perform as well as a Random Forest, mean $R^2$ = 0.27 in test data
# - Feature importance is in agreement with the Random Forest
# 

# ## Hyper parameter tuning 

# The best model is the Random Forest. To ensure the model is not over fit to the training data we compare performance when the following parameters are varied:
# 
# - max_depth: the maximum size of any tree 
# - n_estimators: the number of trees in the forest

# ### Maximum depth

# In[22]:


d = [1,2,3,4,5,6,7]

res_train,res_test = [],[]

for depth in d:
    
    model = RandomForestRegressor(max_depth=depth, 
                                  n_estimators=4, random_state=0)

    y = dta['ae_attendances_attendances']

    X = dta[features]

    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

    scores_train, scores_test = [],[]

    for train_index, test_index in cv.split(X, y):

        model.fit(X.iloc[train_index], y.iloc[train_index])

        scores_test.append(model.score(X.iloc[test_index],
                                       y.iloc[test_index]))
        scores_train.append(model.score(X.iloc[train_index],
                                        y.iloc[train_index]))
        
    res_train.append(scores_train)
    res_test.append(scores_test)


# #### Plot 

# In[23]:


fig,ax = plt.subplots(figsize=(8,5))

plt.plot(d, np.mean(res_train, axis=1), 'b--', label='train')
plt.plot(d, np.mean(res_test, axis=1), 'r--', label='test')

plt.fill_between(d, y1=(np.mean(res_train, axis=1)-np.std(res_train, axis=1)),
                 y2=(np.mean(res_train, axis=1)+np.std(res_train, axis=1)),
                 color='b', alpha=0.2)

plt.fill_between(d, y1=(np.mean(res_test, axis=1)-np.std(res_test, axis=1)),
                 y2=(np.mean(res_test, axis=1)+np.std(res_test, axis=1)),
                 color='r', alpha=0.2)

plt.legend(loc='best')
plt.xlabel('Maximum Tree Depth')
plt.ylabel('Model performance')
plt.show()


# A depth of 5 is optimal. After this, there is no improvement in performance on unseen data (test, red dashed line) and performance continues to increase in the training data (blue dashed line) suggesting overfitting.

# ### Number of trees

# In[24]:


n = [1,2,3,4,5,6,7]

res_train,res_test = [],[]

for est in n:
    
    model = RandomForestRegressor(max_depth=5, n_estimators=est,
                                  random_state=0)

    y = dta['ae_attendances_attendances']

    X = dta[features]

    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

    scores_train, scores_test = [],[]

    for train_index, test_index in cv.split(X, y):

        model.fit(X.iloc[train_index], y.iloc[train_index])

        scores_test.append(model.score(X.iloc[test_index],
                                       y.iloc[test_index]))
        scores_train.append(model.score(X.iloc[train_index],
                                        y.iloc[train_index]))
        
    res_train.append(scores_train)
    res_test.append(scores_test)


# #### Plot 

# In[25]:


fig,ax = plt.subplots(figsize=(8,5))

plt.plot(d, np.mean(res_train, axis=1), 'b--', label='train')
plt.plot(d, np.mean(res_test, axis=1), 'r--', label='test')

plt.fill_between(d, y1=(np.mean(res_train, axis=1)-np.std(res_train, axis=1)),
                 y2=(np.mean(res_train, axis=1)+np.std(res_train, axis=1)),
                 color='b', alpha=0.2)

plt.fill_between(d, y1=(np.mean(res_test, axis=1)-np.std(res_test, axis=1)),
                 y2=(np.mean(res_test, axis=1)+np.std(res_test, axis=1)),
                 color='r', alpha=0.2)

plt.legend(loc='best')
plt.xlabel('Number of Trees')
plt.ylabel('Model Performance')

plt.show()


# The optimal number of trees is 6, beyond which there is no improvement in the training or test set.

# ## Final Model for paper

# Fit the Random forest with optimal parameters

# In[26]:


model = RandomForestRegressor(max_depth=5, n_estimators=6,
                              random_state=0)


features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered']
    
y = dta['ae_attendances_attendances']
X = dta[features]

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
    
scores_train, scores_test, feats = [],[],[]
    
for train_index, test_index in cv.split(X, y):
        
    model.fit(X.iloc[train_index], y.iloc[train_index])
        
    scores_test.append(model.score(X.iloc[test_index],
                                   y.iloc[test_index]))
    scores_train.append(model.score(X.iloc[train_index],
                                    y.iloc[train_index]))

    feats.append(model.feature_importances_)


# ### Performance 

# In[27]:


results=pd.DataFrame()
results['train'] = scores_train
results['test'] = scores_test


# In[28]:


results.describe()


# ### Feature Importance

# In[29]:


feat_imp = pd.DataFrame()

for i,f in enumerate(features):
    
    feat_imp[f] = np.array(feats)[:,i]

feat_imp.describe()


# In[ ]:




