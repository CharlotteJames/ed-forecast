# Forecasting Long Term Emergency Department Demand

The aim of this project is produce long-term forecasts of monthly Emergency Department (ED) attendances in the UK under different scenarios. The majority of existing work on forecasting ED demand uses time series analysis to predict short-term (next day/week) attendances. 
Time-series models assume the past predicts the future, require a large amount of longitudinal data and often don't account for environmental changes. In this context, environmental changes might  be: changes in the way people interact with different components of the health service (utility); changes in the availability of components of the health service (capacity); improvements or declines in overall levels of population health.  

The model presented here is designed to predict mean monthly ED attendances for Clinical Comissioning Groups (CCGs) in the UK using publicly available data. The model has been used to forecast future monthly ED attendances under different scenarios: increases in the capacity of other services and imporvements in overall population health.

While the model in its current form can produce forecasts for individual CCGs, if it were to be used at a regional level it would significantly benefit from being re-trained at a lower level of geography (e.g. LSOA) so that the forecasts are tailored to the regional population.

As the data used is publicly available, when vieving pages with code the notebooks can be run on Google Colab by clicking on the rocket icon. 
