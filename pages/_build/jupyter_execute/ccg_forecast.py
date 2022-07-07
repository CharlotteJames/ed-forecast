#!/usr/bin/env python
# coding: utf-8

# # Future demand at CCG level 

# In[1]:


#turn warnings off to keep notebook tidy
import warnings
warnings.filterwarnings('ignore')


# ## Run forecasting notebook 

# In[2]:


get_ipython().run_line_magic('run', './stacked_forecast.ipynb')


# In[3]:


results


# ## Plot for CCG

# In[4]:


results.ccg.unique()


# ### 26A 

# In[5]:


ccg = '26A'


# In[6]:


fig,ax = plt.subplots(figsize=(8,5))

scenarios = ['Population growth','111 capacity increase',
            'Ambulance capacity increase','GP capacity increase', 'Health of 2019']

for i,results in enumerate(scenario_results):
    
    if i==0:
        
        continue
        
    results_ccg = results.loc[results.ccg==ccg]
    
    points = sum_by_month(results_ccg)
    
    points_series = pd.Series(points)
    
    plt.plot(points_series.rolling(window=6).mean().to_list()[:], label = f'{scenarios[i]}')
    
    
points = sum_by_month(scenario_results[0].loc[scenario_results[0].ccg==ccg])

points_series = pd.Series(points)
    
plt.plot(points_series.rolling(window=6).mean().to_list()[:], 'g--', label = f'{scenarios[0]}')

    
y = np.arange(23500,28000,1000)    
#plt.plot(12*np.ones(len(y)),y, 'k--')

plt.legend(loc = 'best', fontsize=12)
plt.ylabel('Mean monthly ED attendances ', fontsize=14)
plt.xlabel('Months since January 2019', fontsize=14)
plt.xlim(0,60)
#plt.ylim(24000,27000)
plt.tight_layout()

plt.show()


# ### 00Q 

# In[7]:


ccg = '00Q'


# In[8]:


fig,ax = plt.subplots(figsize=(8,5))

scenarios = ['Population growth','111 capacity increase',
            'Ambulance capacity increase','GP capacity increase', 'Health of 2019']

for i,results in enumerate(scenario_results):
    
    if i==0:
        
        continue
        
    results_ccg = results.loc[results.ccg==ccg]
    
    points = sum_by_month(results_ccg)
    
    points_series = pd.Series(points)
    
    plt.plot(points_series.rolling(window=6).mean().to_list()[:], label = f'{scenarios[i]}')
    
    
points = sum_by_month(scenario_results[0].loc[scenario_results[0].ccg==ccg])

points_series = pd.Series(points)
    
plt.plot(points_series.rolling(window=6).mean().to_list()[:], 'g--', label = f'{scenarios[0]}')

    
y = np.arange(23500,28000,1000)    
#plt.plot(12*np.ones(len(y)),y, 'k--')

plt.legend(loc = 'best', fontsize=12)
plt.ylabel('Mean monthly ED attendances', fontsize=14)
plt.xlabel('Months since January 2019', fontsize=14)
plt.xlim(0,60)
#plt.ylim(24000,27000)
plt.tight_layout()

plt.show()


# ### 72Q 

# In[9]:


ccg = '72Q'


# In[10]:


fig,ax = plt.subplots(figsize=(8,5))

scenarios = ['Population growth','111 capacity increase',
            'Ambulance capacity increase','GP capacity increase', 'Health of 2019']

for i,results in enumerate(scenario_results):
    
    if i==0:
        
        continue
        
    results_ccg = results.loc[results.ccg==ccg]
    
    points = sum_by_month(results_ccg)
    
    points_series = pd.Series(points)
    
    plt.plot(points_series.rolling(window=6).mean().to_list()[:], label = f'{scenarios[i]}')
    
    
points = sum_by_month(scenario_results[0].loc[scenario_results[0].ccg==ccg])

points_series = pd.Series(points)
    
plt.plot(points_series.rolling(window=6).mean().to_list()[:], 'g--', label = f'{scenarios[0]}')

    
y = np.arange(23500,28000,1000)    
#plt.plot(12*np.ones(len(y)),y, 'k--')

plt.legend(loc = 'best', fontsize=12)
plt.ylabel('Mean monthly ED attendances', fontsize=14)
plt.xlabel('Months since January 2019', fontsize=14)
plt.xlim(0,60)
#plt.ylim(24000,27000)
plt.tight_layout()

plt.show()


# In[ ]:




