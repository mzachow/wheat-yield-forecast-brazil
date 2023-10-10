# wheat-yield-forecast-brazil

wheat-yield-forecast-brazil is a project containing modules and notebooks to reproduce the findings from the paper [Seasonal climate models for national wheat yield forecasts in Brazil](https://doi.org/10.1016/j.agrformet.2023.109753). The objective of this research was to forecast trend-corrected wheat yield in Brazil before harvest using forecasted climate features from seasonal climate models. 

## Results

Our proposed approach shows a 12% RMSE in forecasting yield early in the season, from April to June. Forecasts start to improve from July onwards, with shorter lead times and including observed features from September on. At the beginning of October, about two months before harvest is completed, wheat yield can be forecasted with 7.6%, 7.9%, 7.9%, 9.1%, and 9.3% RMSE with data from UKMO, ECMWF, MME, NCEP, and CLIMATE respectively. Seasonal climate models are useful tools to forecast national wheat yield, even just before harvest to prepare for food shortages. Our approach could be applied to other staple crops and regions.

## Usage

Preprocessed hindcast data as csv from ECMWF, NCEP, and NCEP will be made available upon request.
To reproduce the insights, there are three notebooks.

1. ``prepare_wheat_data``: read, detrend, and save historical wheat yield data from Brazil. 
2. ``climatology_vs_seasonal_forecasts``: main part of the research with detailed steps and descriptions on how the model was built using wheat and (forecasted and observed) climate data. Here, the preprocessed hindcasts are required.
3. ``figures_for_manuscript``: design of the figures, based on the results of 2.). Figures from the manuscript are saved in /images.


## Acknowledgment

The research was conducted at the [chair of digital agriculture](https://www2.ls.tum.de/dag/startseite/) at Technical University of Munich together with Prof. Dr. Senthold Asseng and Rogério de Souza Nóia Júnior (PhD student). The model described here was based on a previously developed wheat yield estimation model from [Nóia Júnior et al. (2021)](https://iopscience.iop.org/article/10.1088/1748-9326/ac26f3). 

Seasonal climate forecasts were obtained from the [Climate Data Store](https://cds.climate.copernicus.eu/cdsapp#!/home) from the Copernicus Climate Change Service, operated by the [ECMWF](https://confluence.ecmwf.int/display/CKB/Description+of+the+C3S+seasonal+multi-system). The used dataset was [Seasonal forecast daily and subdaily data on single levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-original-single-levels?tab=overview).

Seasonal climate models were bias-adjusted with scaled distribution mapping using code from [bias_correction](https://github.com/pankajkarman/bias_correction)

## License

[GNU GPLv3.0](https://choosealicense.com/licenses/gpl-3.0/)
