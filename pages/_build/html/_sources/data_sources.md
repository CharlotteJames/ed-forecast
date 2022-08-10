# Data Sources

## ED Attendances
ED attendance data was obtained from the A&E Attendances and Emergency Admissions dataset published by NHS England {cite}`AE`. This data is released monthly and contains the total number of attendances at each NHS Trust for that month. NHS Trusts were mapped to CCGs. ED attendances per CCG were calculated as the sum of the attendances at each Trust.

## GP capacity
Data on the number of GP appointments available each CCG was obtained from the Appointments in General Practice dataset, published by NHS Digital {cite}`GP`. This data is released monthly. For each CCG, the variable ‘count of appointments’ was summed to give a total number of appointments available each month as a measure of GP capacity.

## 111 capacity
The NHS 111 minimum dataset was published monthly by NHS England up until March 2021 {cite}`111`. It contains detailed information on call handling and disposition after triage for all 111 contracts covering England. From this dataset, the variable “Calls offered” was used as a measure of the capacity of the 111 service. The number of calls offered was summed for each contract and each month before being stratified by CCG.

## Ambulance capacity
NHS England publish the Ambulance System Indicators (AmbSys) data monthly which contains information on calls made to the 11 ambulance service contracts covering England {cite}`Amb`. We used “Calls answered (A1)” as a measure of service capacity. The number of calls answered was summed for each contract and each month before being stratified by CCG.

## Population
Estimates of the population of CCGs were obtained from the Office for National Statistics (ONS) {cite}`pop`. Estimates of future population were also obtained to allow for forecasting of future ED demand.

## Health Index
The Health Index is a measure of the health of the United Kingdom, consisting of three domains: People (health outcomes), Places (social and environmental risk factors) and Lives (individual risk factors) {cite}`HI_dev`. Values of the health index are relative to index values in 2015. It has a base of 100: values higher than 100 indicate better health than in 2015 and values lower than 100 indicate worse health. The Health Index is an experimental statistic developed by the ONS and available stratified by CCG {cite}`HI`.

```{bibliography}
:style: unsrt
```
