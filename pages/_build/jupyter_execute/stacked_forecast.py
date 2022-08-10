#!/usr/bin/env python
# coding: utf-8

# # Forecasting Future ED Demand 

# ## Overview 

# This notebook contains the code to forecast future ED demand under 4 different scenarios:
# 
# 1. 'do nothing': population grows but population health and health service capacity reamin unchanged
# 2. Increase 111 capacity by 10% in 2020
# 3. Increase 999 capacity by 10% in 2020
# 4. Increase GP capacity by 10% in 2020
# 5. If population health measures (People, Places, Lives) are less than the 2019 average, increase them by 0.2 points per year until they reach the 2019 average. 

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


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# ## Import trained models 

# In[3]:


with open('stacked_model_scaled.pkl','rb') as f:
    
    models, m1_features, m2_features = pkl.load(f)


# **NB** if running notebook on colab the above code wont work. 
# Instead, run the following cell: 

# In[4]:


get_ipython().run_line_magic('run', 'stacked_model.ipynb')

models = [rf1,rf2,final]


# ## Import population forecasts 

# In[5]:


population = pd.read_csv('https://raw.githubusercontent.com/CharlotteJames/ed-forecast/main/data/pop_forecasts_scaled_new.csv',
                  index_col=0)


# In[6]:


population


# ## Import 2019 data as baseline 

# In[7]:


baseline = pd.read_csv('https://raw.githubusercontent.com/CharlotteJames/ed-forecast/main/data/master_scaled_2019_new.csv',
                  index_col=0)


# In[8]:


baseline.columns = ['_'.join([c.split('/')[0],c.split('/')[-1]]) 
                    if '/' in c else c for c in baseline.columns]


# In[9]:


baseline


# ## Functions 

# Model predicts monthly ED attendances per 10,000 people
# 
# To forecast raw numbers, need to multiply predicted value by population/10,000

# In[10]:


def stacked_predict(X, models, m1_features, m2_features):
    
    rf1,rf2,final = models
    
    y_pred_1 = rf1.predict(X[m1_features])

    y_pred_2 = rf2.predict(X[m2_features])

    X_f = np.vstack([y_pred_1, y_pred_2]).T
    
    preds = final.predict(X_f)
    
    return preds


# In[11]:


def forecast(data, pop, year, models, m1_features, m2_features):
    
    # model = reg
    
    data = data.merge(pop[[str(year),'ccg']], 
                      left_on = 'ccg', right_on='ccg')
    
    data['population'] = data[str(year)]#*10000
    
    X = data.drop(['ae_attendances_attendances','ccg',                   'month',str(year),'ccg'], axis=1)
    
    preds = stacked_predict(X, models, m1_features, m2_features)
    
    preds = preds*data['population'].values
    
    return preds


# In[12]:


def sum_by_month(results):

    to_plot = []

    months = ['Jan','Feb','Mar','Apr','May','Jun',              'Jul','Aug','Sep','Oct','Nov','Dec']

    for month in months:

        res = results.loc[results.month==month]
        to_plot.append(np.mean(res[res.columns[2:]].values, axis=0))

    to_plot = np.array(to_plot).T

    points = []

    for row in to_plot:

        points.extend(row)
        
    return points


# ### List to store scenario results 

# In[13]:


scenario_results = []


# ### Scaling factor for capacity increase 

# In[14]:


F=1.1


# ## Scenario 1: do nothing 

# In[15]:


results = pd.DataFrame()
results['ccg'] = baseline['ccg']
results['month'] = baseline['month']
results['2019'] = baseline['ae_attendances_attendances']*baseline['population']

for year in np.arange(2020,2028):

    preds = forecast(baseline,population,year,models,m1_features,m2_features)
    
    results[str(year)] = preds


# In[16]:


points = sum_by_month(results)

fig,ax = plt.subplots(figsize=(6,4))

plt.plot(points[2:])

plt.show()


# In[17]:


scenario_results.append(results)


# ## Scenario 2: increase 111 capacity

# In[18]:


results = pd.DataFrame()
results['ccg'] = baseline['ccg']
results['month'] = baseline['month']
results['2019'] = baseline['ae_attendances_attendances']*baseline['population']

dta = baseline.copy()

for year in np.arange(2020,2028):

    dta['111_111_offered'] = baseline['111_111_offered'].values*F

    preds = forecast(dta,population,year,models,m1_features,m2_features)
    
    results[str(year)] = preds


# In[19]:


points = sum_by_month(results)

fig,ax = plt.subplots(figsize=(6,4))

plt.plot(points[2:])

plt.show()


# In[20]:


scenario_results.append(results)


# ## Scenario 3: increase 999 capacity

# In[21]:


results = pd.DataFrame()
results['ccg'] = baseline['ccg']
results['month'] = baseline['month']
results['2019'] = baseline['ae_attendances_attendances']*baseline['population']

dta = baseline.copy()

for year in np.arange(2020,2028):
    
    dta['amb_sys_answered'] = baseline['amb_sys_answered'].values*F

    preds = forecast(dta,population,year,models,m1_features,m2_features)
    
    results[str(year)] = preds


# In[22]:


points = sum_by_month(results)

fig,ax = plt.subplots(figsize=(6,4))

plt.plot(points[2:])

plt.show()


# In[23]:


scenario_results.append(results)


# ## Scenario 4: increase GP capacity

# In[24]:


results = pd.DataFrame()
results['ccg'] = baseline['ccg']
results['month'] = baseline['month']
results['2019'] = baseline['ae_attendances_attendances']*baseline['population']

dta = baseline.copy()

for year in np.arange(2020,2028):
    
    dta['gp_appt_available'] = baseline['gp_appt_available'].values*F

    preds = forecast(dta,population,year,models,m1_features,m2_features)
    
    results[str(year)] = preds


# In[25]:


points = sum_by_month(results)

fig,ax = plt.subplots(figsize=(6,4))

plt.plot(points[2:])

plt.show()


# In[26]:


scenario_results.append(results)


# ## Scenario 5: health of population at 2019

# In[27]:


results = pd.DataFrame()
results['ccg'] = baseline['ccg']
results['month'] = baseline['month']
results['2019'] = baseline['ae_attendances_attendances']*baseline['population']

dta = baseline.copy()


for year in np.arange(2020,2028):
    
    dta['People'] = [p+0.2 if p<np.mean(baseline.People.values) else p for p in dta.People.values]
    dta['Places'] = [p+0.2 if p<np.mean(baseline.Places.values) else p for p in dta.Places.values]
    dta['Lives'] = [p+0.2 if p<np.mean(baseline.Lives.values) else p for p in dta.Lives.values]

    preds = forecast(dta,population,year,models,m1_features,m2_features)
    
    results[str(year)] = preds


# In[28]:


points = sum_by_month(results)

fig,ax = plt.subplots(figsize=(6,4))

plt.plot(points[2:])

plt.show()


# In[29]:


scenario_results.append(results)


# ## Plot 

# In[30]:


fig,ax = plt.subplots(figsize=(8,5))

scenarios = ['Population growth','111 capacity increase',
            'Ambulance capacity increase','GP capacity increase', 'Health of 2019']

for i,results in enumerate(scenario_results):
    
    if i==0:
        
        continue
    
    points = sum_by_month(results)
    
    points_series = pd.Series(points)
    
    plt.plot(np.arange(-12, 96),
             points_series.rolling(window=4).mean().to_list()[:], 
             label = f'{scenarios[i]}')
    
    
points = sum_by_month(scenario_results[0])

points_series = pd.Series(points)
    
plt.plot(np.arange(-12, 96),
         points_series.rolling(window=4).mean().to_list()[:],
         'g--', label = f'{scenarios[0]}')

    
y = np.arange(24000,29000,1000)    
plt.plot(np.zeros(len(y)),y, 'k--')

plt.legend(loc = 'lower left', fontsize=12)
plt.ylabel('Mean monthly ED attendances per CCG', fontsize=14)
plt.xlabel('Months since baseline', fontsize=14)
plt.xlim(0,48)

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(-12, 50, 10))

#plt.ylim(24000,27000)
plt.tight_layout()
plt.savefig('forecast_scaled.png')
plt.show()


# In[ ]:




