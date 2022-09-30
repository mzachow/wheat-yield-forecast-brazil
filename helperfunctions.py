"""Libraries"""
import glob
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import helperfunctions as hf
import matplotlib.pyplot as plt
from itertools import groupby
from copy import copy, deepcopy
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, SequentialFeatureSelector
from bias_correction import BiasCorrection
from IPython.core.interactiveshell import InteractiveShell

"""Variables"""
crop_seasons = list(range(1993,2017))
months_of_crop_season = list(range(8,11))
homogeneous_groups = list(range(1,5))


# Read Data

def read_raw_model_data():
    """Reads and returns raw hindcasts from ECMWF, UKMO, NCEP and MME as one dataframe."""
    
    ukmo = pd.read_csv("Data/Raw Hindcasts as CSV/ukmo.csv", dtype={"date":str, "group":int}, parse_dates=["date"])
    ncep = pd.read_csv("Data/Raw Hindcasts as CSV/ncep.csv", dtype={"date":str, "group":int}, parse_dates=["date"])
    ecmwf = pd.read_csv("Data/Raw Hindcasts as CSV/ecmwf.csv", dtype={"date":str, "group":int}, parse_dates=["date"])
    
    df = pd.concat([ukmo, ncep, ecmwf])
    df = df.sort_values(by=["model", "init_month", "ensemble", "group", "year", "month", "date"])
    df = df.loc[(df["init_month"] >= 4) & (df["month"] >= 8)].reset_index(drop=True)
    
    ensemble_aggregation = (df
                            .groupby(["model", "init_month", "group", "year", "month", "date"])
                            .agg({"tmean":"mean", "tmax":"mean", "tmin":"mean", "rain":"mean"})
                            .reset_index(drop=False))

    multi_model_ensemble = (df
                            .groupby(["init_month", "group", "year", "month", "date"])
                            .agg({"tmean":"mean", "tmax":"mean", "tmin":"mean", "rain":"mean"})
                            .reset_index(drop=False)
                            .assign(model="MME")
                            .loc[:,ensemble_aggregation.columns])

    hindcasts = (pd.concat([ensemble_aggregation, multi_model_ensemble])
                 .rename(columns={"date":"time", "group":"zone"})
                 .set_index(["model", "init_month", "zone", "year", "month"])
                 .sort_index())
    
    return hindcasts


def read_observed_weather():
    """Reads and returns daily weather observations as dataframe."""
    
    weather_station_to_group_id = {"PFUN":1, "LOND":2, "CAMP":3, "PGRO":4}         
    all_files = glob.glob("Data/Observed Weather/*.csv")
    li = []
    for _, filename in enumerate(all_files):
        observations = pd.read_csv(filename,
                                   usecols=["date", "rain", "tmax", "tmin", "tmean", "treatment"], 
                                   dtype={"date":str}, 
                                   parse_dates=["date"])
        li.append(observations)
        
    observations = (pd
                    .concat(li, axis=0, ignore_index=False)
                    .assign(
                        model="WS", 
                        init_month=11, 
                        year=lambda x: x["date"].dt.year, 
                        month=lambda x: x["date"].dt.month)
                   )
    observations = observations.loc[(observations["month"].isin(months_of_crop_season))].reset_index(drop=True)
    
    observations["zone"] = observations["treatment"].apply(lambda x: weather_station_to_group_id[x])
    observations = (observations
                    .loc[:, ["model", "init_month", "zone", "year", "month", "date", "tmean", "tmax", "tmin", "rain"]]
                    .rename(columns={"date":"time"})
                    .set_index(["model", "init_month", "zone", "year", "month"])
                    .sort_index()
         )
    observations.loc[:, "tmean"] = observations.loc[:, "tmean"].fillna(observations.loc[:, "tmean"].mean())
    observations.loc[:, "tmax"] = observations.loc[:, "tmax"].fillna(observations.loc[:, "tmax"].mean())
    observations.loc[:, "tmin"] = observations.loc[:, "tmin"].fillna(observations.loc[:, "tmin"].mean())

    return observations

def create_climatology_data(observed):
    climatology = (observed
                 .loc[("WS", 11, [1, 2, 3, 4], list(range(1961,1993)), months_of_crop_season)]
                 .groupby(["zone", "year", "month"])
                 .agg({"tmean":"mean", "tmax":"mean", "tmin":"mean", "rain":"sum"})
                 .reset_index()
                 .groupby(["zone", "month"])
                 .agg({"tmean":"mean", "tmax":"mean", "tmin":"mean", "rain":"mean"})
                 .reset_index()
                 .copy())
    return climatology

# Bias Adjustment

def adjust_mean_temperature_bias(observed, predicted, correction_method="normal_mapping"):
    """Return bias-adjusted hindcasts as Dataframe."""
    
    # GroupBy objects allow faster access to relevant subsets of climate data.
    grouped_climate_observations = observed.groupby(["zone", "year", "month"])
    grouped_climate_hindcasts = predicted.groupby(["model", "init_month", "zone", "year", "month"])
    grouped_climate_hindcasts_reference = copy(grouped_climate_hindcasts)
    li = []
    
    for group_idx, group_content in grouped_climate_hindcasts:
        # Save group characteristics in intuitive variables.
        current_model = group_content.index.get_level_values("model")[0]
        current_init_month = group_content.index.get_level_values("init_month")[0]
        current_zone = group_content.index.get_level_values("zone")[0]
        current_season = group_content.index.get_level_values("year")[0]
        current_month = group_content.index.get_level_values("month")[0]
 
        # Create calibration set of observations and hindcasts.
        hindcasts_used_as_reference = []
        observations_used_as_reference = []
        for season in crop_seasons: 
            if season != current_season:
                observation_to_be_added = (current_zone, season, current_month) 
                observations_used_as_reference.append(grouped_climate_observations.get_group(observation_to_be_added))   
            hindcast_to_be_added = (current_model, current_init_month, current_zone, season, current_month)
            hindcasts_used_as_reference.append(grouped_climate_hindcasts_reference.get_group(hindcast_to_be_added))
        hindcasts_used_as_reference = pd.concat(hindcasts_used_as_reference, axis=0, ignore_index=False)
        observations_used_as_reference = pd.concat(observations_used_as_reference, axis=0, ignore_index=False) 
        
        # Perform bias-adjustment for temperature variables.
        bc_tmean = BiasCorrection(observations_used_as_reference["tmean"], hindcasts_used_as_reference["tmean"], group_content["tmean"])
        bc_tmax = BiasCorrection(observations_used_as_reference["tmax"], hindcasts_used_as_reference["tmax"], group_content["tmax"])
        bc_tmin = BiasCorrection(observations_used_as_reference["tmin"], hindcasts_used_as_reference["tmin"], group_content["tmin"])
        group_content["tmean"] = bc_tmean.correct(method=correction_method)
        group_content["tmax"] = bc_tmax.correct(method=correction_method)
        group_content["tmin"] = bc_tmin.correct(method=correction_method)
        
        li.append(group_content)
        
    result = pd.concat(li, axis=0, ignore_index=False)
    return result
            

# Dataset Completion

def fill_missing_dates_with_observations(observations, model):
    """Supplements past days before day of initialization in init_month with weather observations."""
    
    model = model.copy()
    grouped_model_output = model.groupby(["model", "init_month", "zone", "year"])
    li=[]
    for group_characteristics, group_content in grouped_model_output:
        current_model = group_content.index.get_level_values("model")[0]
        current_init_month = group_content.index.get_level_values("init_month")[0]
        current_zone = group_content.index.get_level_values("zone")[0]
        current_season = group_content.index.get_level_values("year")[0]
        
        observations_for_zone_and_season = (observations.loc[("WS", 11, current_zone, current_season)]
                                            .assign(init_month=current_init_month, model=current_model)
                                            .set_index(["model", "init_month"], append=True))
        hindcasts_on_observations = observations_for_zone_and_season.merge(group_content, on="time", 
                                                                           how="left", suffixes=("_ws", "_bcm"))
        hindcasts = hindcasts_on_observations.loc[:,[c for c in hindcasts_on_observations.columns if "_ws" not in c]]
        hindcasts.columns = hindcasts.columns.str.rstrip("_bcm")
        hindcasts = hindcasts.set_index("time")
        observations_for_zone_and_season = observations_for_zone_and_season.set_index("time")
        combined = hindcasts.combine_first(observations_for_zone_and_season)
        combined = (combined.reset_index(drop=False)
                    .assign(model=current_model, init_month=current_init_month, 
                            zone=current_zone, year=current_season, month=lambda x: x["time"].dt.month)
                    .set_index(["model", "init_month", "zone", "year", "month"]))
        li.append(combined)
        if current_init_month == 10:
            fully_observed = (observations_for_zone_and_season
                              .reset_index(drop=False)
                              .assign(model=current_model, init_month=11, 
                                      zone=current_zone, year=current_season, month=lambda x:x["time"].dt.month)
                              .set_index(["model", "init_month", "zone", "year", "month"]))
            li.append(fully_observed)
    result = pd.concat(li, axis=0, ignore_index=False).sort_index()
    
    return result

# Feature Calculation

def aggregate_data(model):
    """Compute monthly climate indices."""
    
    month_conversion = {8:"Aug", 9:"Sep", 10:"Oct"} 
    climate_data_grouped = model.groupby(["model", "init_month", "zone", "year", "month"])
    
    li = []
    for group_characteristics, group_content in climate_data_grouped:
        group_content = (group_content
                         .groupby(["model", "init_month", "zone", "year", "month"])
                         .agg({"tmean":"mean", "tmax":"mean", "tmin":"mean", "rain":"sum"})
                         .reset_index())
        li.append(group_content)           

    monthly_indices = pd.concat(li, axis=0, ignore_index=False)
    monthly_indices.columns = ["model", "init_month", "zone", "year", "month", "Tmean", "Tmax", "Tmin", "Rain"]
    monthly_indices["month"] = monthly_indices["month"].replace(month_conversion) 
    monthly_indices = (monthly_indices
                       .reset_index(drop=True)
                       .pivot(index=["model", "init_month", "zone", "year"], columns="month"))
    monthly_indices.columns = [str(s[0]) + "_" + str(s[1]) for s in monthly_indices.columns]
    monthly_indices = monthly_indices.reset_index().sort_values(by=["model", "init_month", "zone", "year"])
    
    return monthly_indices

def create_climatology_features(features, climate):
    climate = climate.copy()
    features = features.loc[features["model"] == "WS"].reset_index(drop=True).copy()
    month_conversion = {8:"Aug", 9:"Sep", 10:"Oct"} 
    climate.columns = ["zone", "month", "Tmean", "Tmax", "Tmin", "Rain"]
    climate["month"] = climate["month"].replace(month_conversion) 
    climate["model"] = "CLIMATE"
    climate = (climate
               .reset_index(drop=True)
               .pivot(index=["model", "zone"], columns="month"))
    climate.columns = [str(s[0]) + "_" + str(s[1]) for s in climate.columns]
    climate = climate.reset_index()
    climate = pd.concat([climate]*8)
    init_months = np.repeat([4, 5, 6, 7, 8, 9, 10, 11], 4)
    climate["init_month"] = init_months
    climate = pd.concat([climate]*24).reset_index(drop=True)
    climate["year"] = np.repeat(list(range(1993,2017)), 32)
    climate = climate.set_index(["model", "init_month", "zone", "year"]).sort_index().reset_index()
    li = []
    for im in list(range(4,12)):
        temp = climate.loc[climate["init_month"] == im].copy()
        if im <= 8:
            li.append(temp)
        if im == 9:
            temp.loc[:, [c for c in temp.columns if ("Aug" in c)]] = np.nan
            li.append(temp)
        if im == 10:
            temp.loc[:, [c for c in temp.columns if ("Sep" in c) | ("Aug" in c)]] = np.nan
            li.append(temp)
        if im == 11:
            temp.loc[:, [c for c in temp.columns if ("Oct" in c) | ("Sep" in c) | ("Aug" in c)]] = np.nan
            li.append(temp)
    climate = pd.concat(li, axis=0).set_index(["zone", "year"])
    features = features.set_index(["zone", "year"])
    climate = climate.combine_first(features).reset_index()
    
    return climate


# Yiel Data
def read_national_wheat_yield():
    national_yield = pd.read_csv("Data/Wheat/ibge_national_yield_detrended.csv")
    return national_yield

# K-Fold Cross Validation
def kfold_cross_validation(data, model="ECMWF", init=8, no_of_features=8):
    """
    Returns predictions on LOO-CV.
        Params:
            data, dataframe: all features and targets by group and year for all models
            group, int: group to perform forecasts on
            model, string: model that is evaluated
            init, int: init_month that is evaluated
        Returns:
            result, dataframe: national yield forecasts by year
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
