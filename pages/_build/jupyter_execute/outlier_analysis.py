#!/usr/bin/env python
# coding: utf-8

# # Outlier analysis 

# When plotting true vs predicted ED attendances (see [Combined Population Health and Service Capacity Model](./stacked_model.ipynb) and below) there are two clusters of points that appear to be outliers: Liverpool and Hull.
# 
# To understand why these points are outliers in the following we:
# 
# - fit the model without these points and assess change in performance
# - investigate the relationship between population health variables and mean monthly ED attendances for these points

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


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score as r2


import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# ## Import data 

# In[3]:


dta = pd.read_csv('../data/master_scaled_new.csv', index_col=0)


# In[4]:


dta.columns = ['_'.join([c.split('/')[0],c.split('/')[-1]])
               if '/' in c else c for c in dta.columns]


# In[5]:


dta.head()


# In[6]:


dta.shape


# ## Function to group data 

# In[7]:


def group_data(data, features):

    features = ['population',
                'People', 'Places',
                'Lives']


    #ensure no identical points in train and test

    grouped = pd.DataFrame()
    
    for pop, group in data.groupby('population'):

        #if len(group.lives.unique())>1:

            #print('multiple CCG with same population')

        ccg_year = pd.Series(dtype='float64')

        for f in features:

            ccg_year[f] = group[f].unique()[0]

        ccg_year['ae_attendances_attendances']        = group.ae_attendances_attendances.mean()
        

        grouped = grouped.append(ccg_year, ignore_index=True)
        
    return grouped


# ## Functions to fit model 

# In[8]:


def fit_ph(dta, features, model):
    
    if 'ae_predicted' in dta.columns:
        
        dta = dta.drop(['ae_predicted'], axis=1)
    
    grouped = group_data(dta, features)
    
    y = grouped['ae_attendances_attendances']

    X = grouped[features]

    # dont set random state so that function can be used in overall cv
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    
    results = pd.DataFrame()
    
    for train_index, test_index in cv.split(X, y):
        
        model.fit(X.iloc[train_index], y.iloc[train_index])
        
        test = X.iloc[test_index].copy()
        
        test['ae_predicted'] = model.predict(X.iloc[test_index])
    
        results = results.append(test, ignore_index=True)
        
    dta = dta.merge(results[['population','ae_predicted']],
                    left_on='population', right_on='population')
        
    return dta


# In[9]:


def fit_capacity(dta, features, model):
    
    y = dta['ae_attendances_attendances']
    X = dta[features]

    model.fit(X,y)
    
    return model


# In[10]:


def fit_combined(train, rf1, m1_features, train_size=7/8):
    
    final = LinearRegression()

    #split training data into two sets
    train_0, train_1 = train_test_split(train, 
                                        train_size=train_size, 
                                        random_state=29)

    #train capactiy model
    
    rf1 = fit_capacity(train_0, m1_features, rf1)
    

    #predict monthly attendances

    y_pred_1 = rf1.predict(train_1[m1_features])

    
    #use pre-predicted average attendances
    
    y_pred_2 = train_1['ae_predicted']
        
    #final
        
    X_f = np.vstack([y_pred_1, y_pred_2]).T
    y_f = train_1['ae_attendances_attendances']

    final.fit(X_f,y_f)
    
    return rf1,final        


# In[11]:


def cv_combined(dta, rf1, rf2):
    
    # splitter for cross validation 
    
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
    
    scores_final, scores_rf1, scores_rf2, coefs = [],[],[],[]
    
    k=1
    
    capacity_features = ['gp_appt_available',
            '111_111_offered', 'amb_sys_answered']

    pophealth_features = ['population',
                            'People', 'Places', 'Lives']
    
    dta_pred = pd.DataFrame()
    
    #fit population health independently to avoid data leakage
    
    dta = fit_ph(dta, pophealth_features, rf2)
    
    print(dta.shape)
    
    for train_index, test_index in cv.split(dta):
        
        #print(f'\n Split {k} \n')
        
        train = dta.iloc[train_index]
        test = dta.iloc[test_index]

        #final models
        rf1, final = fit_combined(train, rf1, capacity_features)
        
        coefs.append(final.coef_)
        
        #predict on test data
        
        y_pred_cu = rf1.predict(test[capacity_features])

        scores_rf1.append(rf1.score(test[capacity_features],
                                    test['ae_attendances_attendances']))

        y_pred_ph = test['ae_predicted']

        scores_rf2.append(r2(test['ae_attendances_attendances'],
                             test['ae_predicted']))
        
        preds = final.predict(np.vstack([y_pred_cu, y_pred_ph]).T)

        scores_final.append(final.score(np.vstack([y_pred_cu, y_pred_ph]).T,
                                        test['ae_attendances_attendances']))
        
        test_pred = test.copy()
        test_pred['predicted'] = preds
        test_pred['true'] = test['ae_attendances_attendances'].values
        
        test_pred['iter'] = [k for i in test_pred.index]
        
        dta_pred = dta_pred.append(test_pred, ignore_index=False)
        
        
        k+=1
        
    return scores_final, scores_rf1, scores_rf2, dta_pred, coefs


# ## Fit model 

# In[12]:


#capacity model
rf1 = RandomForestRegressor(max_depth=5, n_estimators=6, random_state=0)

#population health model
rf2 = RandomForestRegressor(max_depth=5, n_estimators=4, random_state=0)

scores_final, scores_rf1, scores_rf2, dta_pred, coefs = cv_combined(dta, rf1, rf2)


# In[13]:


results=pd.DataFrame()

results['final'] = scores_final


# In[14]:


results.describe()


# ### Coefficient importances 

# #### Mean 

# In[15]:


np.mean(coefs, axis=0)


# #### Std 

# In[16]:


np.std(coefs, axis=0)


# ### Plot 

# In[17]:


fig,ax = plt.subplots(figsize=(8,5))

mean_pred, true = [],[]

for i in dta_pred.index.unique():
    
    mean_pred.append(dta_pred.loc[i]['predicted'].mean())
    true.append(dta_pred.loc[i]['true'].mean())

plt.plot(true, mean_pred, 'o', alpha=0.5)

xx = np.arange(min(dta_pred['true']),max(dta_pred['true']))
plt.plot(xx,xx,'k--')

plt.xlabel('True monthly ED attendances per 10,000 people')
plt.ylabel('Predicted monthly ED attendances per 10,000 people')
plt.savefig('true_predicted_combined.png', dpi=300)
plt.show()


# ## Remove Outliers (Hull and Liverpool) 

# In[18]:


dta.loc[dta.ccg=='03F']#.shape


# In[19]:


dta.loc[dta.ccg=='99A']


# In[20]:


hull = dta.loc[(dta.ccg=='03F') & (dta.year==2018)]
liv = dta.loc[(dta.ccg=='99A') & (dta.year==2019)]


# In[21]:


dta2 = dta.drop(hull.index)
dta2 = dta2.drop(liv.index)

scores_final, scores_rf1, scores_rf2, dta_pred, coefs = cv_combined(dta2, rf1, rf2)


# ### Results

# In[22]:


results=pd.DataFrame()

results['final'] = scores_final


# In[23]:


results.describe()


# ### Plot 

# In[24]:


fig,ax = plt.subplots(figsize=(8,5))

mean_pred, true = [],[]

for i in dta_pred.index.unique():
    
    mean_pred.append(dta_pred.loc[i]['predicted'].mean())
    true.append(dta_pred.loc[i]['true'].mean())

plt.plot(true, mean_pred, 'o', alpha=0.5)

xx = np.arange(min(dta_pred['true']),max(dta_pred['true']))
plt.plot(xx,xx,'k--')

plt.xlabel('True monthly ED attendances per 10,000 people')
plt.ylabel('Predicted monthly ED attendances per 10,000 people')
plt.savefig('true_predicted_combined.png', dpi=300)
plt.show()


# ## Plot coloured by each grouped feature 

# In[25]:


features = ['population',
            'People', 'Places', 'Lives']

grouped = group_data(dta, features)


# In[26]:


grouped


# In[27]:


scores_final, scores_rf1, scores_rf2, dta_pred, coefs = cv_combined(dta, rf1, rf2)


# In[28]:


fig,ax_list = plt.subplots(2,2, figsize=(16,10))

mean_pred, true, col = [],[],[]

feats = ['People','Places','Lives','population']
# convert range into 

for i,f in enumerate(feats):
    
    ax = ax_list.flatten()[i]
    
    for i in dta_pred.index.unique():

        mean_pred.append(dta_pred.loc[i]['predicted'].mean())
        true.append(dta_pred.loc[i]['true'].mean())
        col.append(dta_pred.loc[i][f].mean())


    p = ax.scatter(true, mean_pred,  c = col, cmap='viridis')

    xx = np.arange(min(dta_pred['true']),max(dta_pred['true']))
    ax.plot(xx,xx,'k--')

    ax.set_xlabel('True monthly ED attendances per 10,000 people')
    ax.set_ylabel('Predicted monthly ED attendances per 10,000 people')
    ax.set_title(f'{f}')
    
    fig.colorbar(p, pad=0.05, ax=ax)
    
plt.tight_layout()
    
plt.savefig('true_predicted_grouped.png', dpi=300)
plt.show()


# ## Plot predicted mean monthly attendances 

# In[29]:


fig,ax_list = plt.subplots(2,2, figsize=(16,10))

mean_pred, true, col = [],[],[]

feats = ['People','Places','Lives','population']
# convert range into 

for i,f in enumerate(feats):
    
    ax = ax_list.flatten()[i]
    
    for c in dta_pred.ccg.unique():
        
        for y in dta_pred.year.unique():

            mean_pred.append(dta_pred.loc[(dta_pred.ccg==c) & (dta_pred.year==y)]['ae_predicted'].mean())
            true.append(dta_pred.loc[(dta_pred.ccg==c) & (dta_pred.year==y)]['true'].mean())
            col.append(dta_pred.loc[(dta_pred.ccg==c) & (dta_pred.year==y)][f].mean())


    p = ax.scatter(true, mean_pred,  c = col, cmap='viridis')

    xx = np.arange(min(dta_pred['true']),max(dta_pred['true']))
    ax.plot(xx,xx,'k--')

    ax.set_xlabel('True mean monthly ED attendances per 10,000 people')
    ax.set_ylabel('Predicted mean monthly ED attendances \n per 10,000 people')
    ax.set_title(f'{f}')
    
    fig.colorbar(p, pad=0.05, ax=ax)
    
plt.tight_layout()
    
plt.savefig('true_predicted_mean.png', dpi=300)
plt.show()


# ## Summary 

# From the above figure, the outliers points have low values of 'People' and 'Lives'. When fitting the population health model, as variables only change annually there are fewer datapoints (145 total).
# 
# Looking at the ‘Lives’ panel in the final figure, if the model is trained on a subsection of the data that contains data points with low values of this variable (darker points) but high mean monthly ED attendances (cluster in top right), it will predict that an unseen datapoint with a low value of Lives has a high number of ED attendances (dark point centre top – that’s Hull).
# 
# As the MGSR assigns a higher weight to the population health model, if the prediction of the population health model isn’t as good for a given ccg, that ccg will be an outlier overall. Model perfomance would improve with an increased amount of data in the future.

# In[ ]:




