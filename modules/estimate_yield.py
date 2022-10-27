#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""estimate_yield contains functions to train wheat yield models on cross-validation."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold

crop_seasons = list(range(1993,2017))
months_of_crop_season = list(range(4,11))
homogeneous_groups = list(range(1,5))
month_conversion = {4:"April", 5:"May", 6:"June", 7:"July", 8:"Aug", 9:"Sep", 10:"Oct"} 
vars_group1 = {'Bias':2, 'Tmax_Aug':0.04, 'Ltemp_July':0.05, 'Ltemp_Oct':-0.4, 'Tmean_Oct':-0.07, 'Rain_Sep':-0.0009, 
               'Hrainfall_Aug':0.03, 'Hrainfall_July':-0.06,'Tmin_Sep':0.009, 'Ltemp_May':-0.04,'Tmean_June':-0.06, 
               'Hrainfall_May':0.03, 'Tmin_Aug':-0.06, 'Htemp_Oct':-0.04, 'Ltemp_June':-0.01, 'Rainy_days_Sep':-0.004}
vars_group2 = {'Bias':-0.9, 'Drought_Sep':0.06, 'Ltemp_July':-0.15, 'Rainy_days_June':0.01, 'Rain_Oct':-0.002, 'Drought_May':-0.06, 
               'Drought_Oct':-0.25, 'Tmean_Sep':0.1, 'Rainy_days_Aug':0.05, 'Tmax_Aug':-0.06, 'Ltemp_June':-0.07, 
               'Tmin_June':0.11, 'Drought_July':-0.06, 'Ltemp_Aug':0.5, 'Tmin_Oct':-0.05}
vars_group3 = {'Bias':3.34, 'Tmin_June':0.024, 'Ltemp_Aug':-0.09, 'Tmin_Aug':-0.028, 'Tmax_May':-0.13, 'Rainy_days_May':-0.014, 
               'Ltemp_Sep':-0.15, 'Drought_Sep':-0.057, 'Ltemp_May':0.22, 'Rainy_days_Aug':0.02, 'Hrainfall_Sep':-0.019, 
               'Drought_May':0.018, 'Rainy_days_July':0.006, 'Htemp_Aug':-0.008, 'Hrainfall_July':-0.009}
vars_group4 = {'Bias':1.8, 'Tmin_Oct':-0.05, 'Ltemp_June':0.08, 'Tmin_Sep':-0.04, 'Rain_Oct':-0.001, 
               'Ltemp_Aug':-0.06, 'Tmax_June':0.02, 'Hrainfall_May':-0.04, 'Ltemp_Sep':-0.07, 
               'Tmean_May':-0.03, 'Tmean_Aug':-0.02, 'Hrainfall_Sep':-0.008}
coeffs = [vars_group1, vars_group2, vars_group3, vars_group4] 
contributions_to_national_yield = {1:0.37, 2:0.23, 3:0.23, 4:0.18}


def retrain_weights(data, yield_df, model="ECMWF", init=8):
    """Retrain the weights of the original model from Nóia Júnior et al. (2021).
    
    params:
     - data: dataframe with the features and yield for all years on group level
     - yield_df: datframe with national wheat yield for all years
     - model: hindcast model to validate (default: ECMWF)
     - init: month of model initialization to validate (default:8)
     
    returns:
     - national_forecasts_by_year: dataframe with forecasted and observed national wheat yield for the selected model and month of initialization on cross validation
    """
    # Filter by model and init_month but also include observations that are used for model training
    cv_dataset = (data.loc[(data["model"].isin([model, "WS"])) 
                           & (data["init_month"].isin([init, 11]))])
    # Dataframe where interim results are saved
    national_forecasts_by_year = (pd.DataFrame(data={"year":crop_seasons, "predicted":np.zeros(24)})
                                  .merge(yield_df, on="year", how="left"))
    
    for season in crop_seasons:
        for group in list(range(1,5)):
            parameters_and_coefficients = coeffs[int(group) - 1]
            X_train = cv_dataset.loc[(cv_dataset["model"] == "WS")
                                      & (cv_dataset["zone"] == group)
                                       & (cv_dataset["year"] != season), [c for c in cv_dataset.columns if (c in list(parameters_and_coefficients.keys()))]]
            y_train = cv_dataset.loc[(cv_dataset["model"] == "WS")
                                      & (cv_dataset["zone"] == group)
                                       & (cv_dataset["year"] != season), "yield"]
            pipeline = Pipeline([('scaler', StandardScaler()), 
                                 ('estimator', Ridge())])
            reg = pipeline.fit(X_train, y_train)  
            X_val = cv_dataset.loc[(cv_dataset["model"] == model)
                                    & (cv_dataset["zone"] == group)
                                     & (cv_dataset["year"] == season), [c for c in cv_dataset.columns if (c in list(parameters_and_coefficients.keys()))]].reset_index(drop=True)
                
            y_predicted = reg.predict(X_val)[0]
            
            # each forecast is weighted by the group's relative contribution to national harvested area
            national_forecasts_by_year.loc[national_forecasts_by_year["year"] == season, "predicted"] += y_predicted * contributions_to_national_yield[group]
    return national_forecasts_by_year


def calculate_estimates(data, yield_df, model="ECMWF", init=8, no_of_features=6):
    """Retrain the weights and select features.
    
    params:
     - data: dataframe with the features and yield for all years on group level
     - yield_df: datframe with national wheat yield for all years
     - model: hindcast model to validate (default: ECMWF)
     - init: month of model initialization to validate (default:8)
     - no_of_features: the number of most correlated features with the target to be selected
          
    returns:
     - national_forecasts_by_year: dataframe with forecasted and observed national wheat yield for the selected model and month of initialization on cross validation
    """
    # Filter by model and init_month but also include observations that are used for model training
    cv_dataset = (data.loc[(data["model"].isin([model, "WS"])) 
                           & (data["init_month"].isin([init, 11]))])
    # Dataframe where interim results are saved
    national_forecasts_by_year = (pd.DataFrame(data={"year":crop_seasons, "predicted":np.zeros(24)})
                                  .merge(yield_df, on="year", how="left"))
    # Features
    relevant_columns = [c for c in cv_dataset.columns if c not in ["model", "init_month", "zone", "year", "yield"]]
    
    for season in crop_seasons:
        for group in list(range(1,5)):
            X_train = cv_dataset.loc[(cv_dataset["model"] == "WS")
                                      & (cv_dataset["zone"] == group)
                                       & (cv_dataset["year"] != season), relevant_columns]
            y_train = cv_dataset.loc[(cv_dataset["model"] == "WS")
                                      & (cv_dataset["zone"] == group)
                                       & (cv_dataset["year"] != season), "yield"]
            # To overcome variance threshold
            if model == "CLIMATE": X_train += np.random.normal(0, 1e-6, X_train.shape) 
            
            pipeline = Pipeline([('scaler', StandardScaler()), 
                                 ('var', VarianceThreshold()), 
                                 ('selector', SelectKBest(f_regression, k=no_of_features)),
                                 ('estimator', Ridge())])
            reg = pipeline.fit(X_train, y_train)  
            X_val = cv_dataset.loc[(cv_dataset["model"] == model)
                                    & (cv_dataset["zone"] == group)
                                     & (cv_dataset["year"] == season), relevant_columns].reset_index(drop=True)
                
            y_predicted = reg.predict(X_val)[0]
            
            # each forecast is weighted by the group's relative contribution to national harvested area
            national_forecasts_by_year.loc[national_forecasts_by_year["year"] == season, "predicted"] += y_predicted * contributions_to_national_yield[group]
    return national_forecasts_by_year


# K-Fold Cross Validation
def kfold_cross_validation(data, model="ECMWF", init=8, no_of_features=8):
    """Retrain, select features, and directly forecast yield on national level.
    
    params:
     - data: all features and yield on national level for all years
     - model: hindcast model to validate (default: ECMWF)
     - init: month of model initialization to validate (default:8)
     - no_of_features: the number of most correlated features with the target to be selected
     
    returns:
     - national_forecasts_by_year: dataframe with forecasted and observed national wheat yield for the selected model and month of initialization on cross validation
    """
    # Filter by model and init_month but also include observations that are used for model training
    cv_dataset = (data.loc[((data["model"] == model) & (data["init_month"] == init))
                               | ((data["model"] == "WS") & (data["init_month"] == 11))])
   
    # Dataframe where interim results are saved
    national_forecasts_by_year = (pd.DataFrame(data={"year":crop_seasons, "predicted":np.zeros(24)})
                                  .merge(cv_dataset.loc[(cv_dataset["model"] == "WS"), 
                                                        ["year", "yield"]], on="year", how="left"))
    # Features
    relevant_columns = [c for c in cv_dataset.columns if c not in ["model", "init_month", "year", "yield"]]
    
    for season in crop_seasons:
        X_train = cv_dataset.loc[(cv_dataset["model"] == "WS") 
                                 & (cv_dataset["year"] != season), relevant_columns].reset_index(drop=True)
        y_train = cv_dataset.loc[(cv_dataset["model"] == "WS") 
                                 & (cv_dataset["year"] != season), "yield"].reset_index(drop=True)
            
        pipeline = Pipeline([('scaler', StandardScaler()), 
                             ('var', VarianceThreshold()), 
                             ('selector', SelectKBest(f_regression, k=no_of_features)),
                             ('estimator', Ridge())])
        
        reg = pipeline.fit(X_train, y_train)  

        X_val = cv_dataset.loc[(cv_dataset["model"] == model)
                               & (cv_dataset["year"] == season), relevant_columns].reset_index(drop=True)
                
        y_predicted = reg.predict(X_val)[0]
            
        # each forecast is weighted by the group's relative contribution to national harvested area
        national_forecasts_by_year.loc[national_forecasts_by_year["year"] == season, "predicted"] = y_predicted
    
    return national_forecasts_by_year
