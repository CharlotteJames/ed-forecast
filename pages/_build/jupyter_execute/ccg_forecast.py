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


# ## Plot for CCG

# ### 26A 

# In[3]:


ccg = '26A'


# In[4]:


fig,ax = plt.subplots(figsize=(8,5))

scenarios = ['Population growth','111 capacity increase',
            'Ambulance capacity increase','GP capacity increase', 'Health of 2019']

for i,results in enumerate(scenario_results):
    
    if i==0:
        
        continue
        
    results_ccg = results.loc[results.ccg==ccg]
    
    points = sum_by_month(results_ccg)
    
    points_series = pd.Series(points)
    
    plt.plot(np.arange(-12, 96), 
             points_series.rolling(window=4).mean().to_list()[:], 
             label = f'{scenarios[i]}')
    
    
points = sum_by_month(scenario_results[0].loc[scenario_results[0].ccg==ccg])

points_series = pd.Series(points)
    
plt.plot(np.arange(-12, 96),
         points_series.rolling(window=4).mean().to_list()[:],
         'g--', label = f'{scenarios[0]}')

    
y = np.arange(23500,28000,1000)    
#plt.plot(12*np.ones(len(y)),y, 'k--')

plt.legend(loc = 'best', fontsize=12)
plt.ylabel('Monthly ED attendances', fontsize=14)
plt.xlabel('Months since baseline', fontsize=14)


plt.xlim(0,48)

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(-12, 50, 10))
#plt.ylim(24000,27000)
plt.tight_layout()

plt.show()


# ### 00Q 

# In[5]:


ccg = '00Q'


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
    
    plt.plot(np.arange(-12, 96), 
             points_series.rolling(window=4).mean().to_list()[:], 
             label = f'{scenarios[i]}')
    
    
points = sum_by_month(scenario_results[0].loc[scenario_results[0].ccg==ccg])

points_series = pd.Series(points)
    
plt.plot(np.arange(-12, 96),
         points_series.rolling(window=4).mean().to_list()[:],
         'g--', label = f'{scenarios[0]}')

    
y = np.arange(23500,28000,1000)    
#plt.plot(12*np.ones(len(y)),y, 'k--')

plt.legend(loc = 'best', fontsize=12)
plt.ylabel('Monthly ED attendances', fontsize=14)
plt.xlabel('Months since baseline', fontsize=14)
plt.xlim(0,48)

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(-12, 50, 10))
#plt.ylim(24000,27000)
plt.tight_layout()

plt.show()


# ### 72Q 

# In[7]:


ccg = '72Q'


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
    
    plt.plot(np.arange(-12, 96), 
             points_series.rolling(window=4).mean().to_list()[:], 
             label = f'{scenarios[i]}')
    
    
points = sum_by_month(scenario_results[0].loc[scenario_results[0].ccg==ccg])

points_series = pd.Series(points)
    
plt.plot(np.arange(-12, 96),
         points_series.rolling(window=4).mean().to_list()[:],
         'g--', label = f'{scenarios[0]}')

    
y = np.arange(23500,28000,1000)    
#plt.plot(12*np.ones(len(y)),y, 'k--')

plt.legend(loc = 'best', fontsize=12)
plt.ylabel('Monthly ED attendances', fontsize=14)
plt.xlabel('Months since baseline', fontsize=14)
plt.xlim(0,48)

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(-12, 50, 10))
#plt.ylim(24000,27000)
plt.tight_layout()

plt.show()


# ## Summary 

# It is clear that, when looking at individual CCGs, the forecasts can vary significantly from the mean results presented on the previous page.
# 
# This demonstrates that for each CCG the model is learning different relationships between the different variables.
# 
# Forecasts at the CCG level would be more accurate if local data was fed into the model.

# In[ ]:




