#!/usr/bin/env python
# coding: utf-8

# # Developing the Population Health Model

# ## Overview 

# This notebook contains the code to develop the population health model. 
# 
# Initially 3 different models are compared (Linear regression, Random Forest regresstion, Gradient Boosted regression).
# 
# Hyper-parameters of the best model are fine-tuned to maximise performance in unseen data while preventing over-fitting and minimising model complexity

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


# ### Function to aggregate data  

# In[9]:


def group_data(data, features):

    #ensure no identical points in train and test

    grouped = pd.DataFrame()
    
    for pop, group in data.groupby('population'):

        #if len(group.lives.unique())>1:

        #    print('multiple CCG with same population')

        ccg_year = pd.Series(dtype='float64')

        for f in features:

            ccg_year[f] = group[f].unique()[0]

        ccg_year['ae_attendances_attendances']         = group.ae_attendances_attendances.mean()
        

        grouped = grouped.append(ccg_year, ignore_index=True)
        
    return grouped


# ## Model Comparison

# Features in the dataset that measure population and population health are:
# 
# - *population*: 
# - *People*: 
# - *Places*: 
# - *Lives*: 

# ### Pair plot 

# In[10]:


features = ['population',
            'People', 'Places',
            'Lives', 'rand1']

grouped = group_data(dta, features)

fig = sns.pairplot(grouped.select_dtypes(include=np.number),
                   kind="reg",
                   plot_kws={'line_kws':{'color':'black'}, 
                            'scatter_kws': 
                             {'color':'green','alpha': 0.1}},
                   diag_kws={'color':'red'})


# ### Linear Regression 

# In[11]:


model = LinearRegression()

features = ['population',
            'People', 'Places',
            'Lives', 'rand1']

grouped = group_data(dta, features)
    
y = grouped['ae_attendances_attendances']
X = grouped[features]

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

# In[12]:


results=pd.DataFrame()
results['train'] = scores_train
results['test'] = scores_test


# In[13]:


results.describe()


# #### Feature Importance 

# In[14]:


feat_imp = pd.DataFrame()

for i,f in enumerate(features):
    
    feat_imp[f] = np.array(feats)[:,i]

feat_imp.describe()


# ### Random Forest 

# In[15]:


model = RandomForestRegressor(max_depth=4, n_estimators=2, 
                              random_state=0)

y = grouped['ae_attendances_attendances']

X = grouped[features]

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

# In[16]:


results=pd.DataFrame()
results['train'] = scores_train
results['test'] = scores_test


# In[17]:


results.describe()


# #### Feature Importance 

# In[18]:


feat_imp = pd.DataFrame()

for i,f in enumerate(features):
    
    feat_imp[f] = np.array(feats)[:,i]

feat_imp.describe()


# ### Gradient boosted tress

# In[19]:


model = GradientBoostingRegressor(max_depth=5, n_estimators=5,
                                  random_state=1)

y = grouped['ae_attendances_attendances']

X = grouped[features]

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

# In[20]:


results=pd.DataFrame()
results['train'] = scores_train
results['test'] = scores_test


# In[21]:


results.describe()


# #### Feature Importance 

# In[22]:


feat_imp = pd.DataFrame()

for i,f in enumerate(features):
    
    feat_imp[f] = np.array(feats)[:,i]

feat_imp.describe()


# ### Summary 

# **Logistic Regression**
# 
# - Variable performance with different splits: mean $R^2$ = 0.4, minimum $R^2$ = -0.5 in test set
# 
# **Random Forest**
# 
# - Best performance with mean $R^2$ = 0.6 in test data. 
# - Performance also variable: minimum $R^2$ = 0.1
# - Feature importance is stable: population is most important, followed by Lives, People then Places. 
# - The random feature has low importnace which validates the importance of other features.
# 
# **Gradient Boosted Trees**
# 
# - Doesn't perform as well as a Random Forest, mean $R^2$ = 0.4 in test data
# - Performance also variable: minimum $R^2$ = 0.04
# - Feature importance is not agreement with the Random Forest, with Places more important than People.
# 

# ## Hyper parameter tuning 

# The best model is the Random Forest. To ensure the model is not over fit to the training data we compare performance when the following parameters are varied:
# 
# - max_depth: the maximum size of any tree 
# - n_estimators: the number of trees in the forest

# ### Maximum Depth 

# In[23]:


d = [1,2,3,4,5,6,7]

res_train,res_test = [],[]

for depth in d:
    
    model = RandomForestRegressor(max_depth=depth, 
                                  n_estimators=3, random_state=0)

    y = grouped['ae_attendances_attendances']

    X = grouped[features]

    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

    scores_train, scores_test = [],[]

    for train_index, test_index in cv.split(X, y):

        model.fit(X.iloc[train_index], y.iloc[train_index])

        scores_test.append(model.score(X.iloc[test_index],
                                       y.iloc[test_index]))
        scores_train.append(model.score(X.iloc[train_index]
                                        , y.iloc[train_index]))
        
    res_train.append(scores_train)
    res_test.append(scores_test)


# #### Plot 

# In[24]:


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


# A depth of 3 is optimal. After this, there is no improvement in performance on unseen data (test, red dashed line) and performance continues to increase in the training data (blue dashed line) suggesting overfitting.

# ### Number of Trees

# In[25]:


n = [1,2,3,4,5,6,7]

res_train,res_test = [],[]

for est in n:
    
    model = RandomForestRegressor(max_depth=3, n_estimators=est,
                                  random_state=0)

    y = grouped['ae_attendances_attendances']

    X = grouped[features]

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

# In[26]:


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
plt.ylabel('Model performance')

plt.show()


# The optimal number of trees is 2, beyond which there is no improvement in the training or test set.

# ## Final Model

# Fit the Random forest with optimal parameters

# In[27]:


model = RandomForestRegressor(max_depth=3, n_estimators=2,
                              random_state=0)

y = grouped['ae_attendances_attendances']

X = grouped[features]

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
    
scores_train, scores_test, feat = [],[],[]
    
for train_index, test_index in cv.split(X, y):
        
    model.fit(X.iloc[train_index], y.iloc[train_index])
        
    scores_test.append(model.score(X.iloc[test_index],
                                   y.iloc[test_index]))
    scores_train.append(model.score(X.iloc[train_index],
                                    y.iloc[train_index]))
    
    feat.append(model.feature_importances_)


# ### Performance 

# In[28]:


results=pd.DataFrame()
results['train'] = scores_train
results['test'] = scores_test


# In[29]:


results.describe()


# ### Feature Importance 

# In[30]:


feat_imp = pd.DataFrame()

for i,f in enumerate(features):
    
    feat_imp[f] = np.array(feats)[:,i]

feat_imp.describe()

