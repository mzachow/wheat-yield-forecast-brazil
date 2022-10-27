#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Preprocess_data contains functions to prepare the data for the wheat yield model.

Functionaly includes:
 - bias-adjustment
 - feature aggregation
 - ..
"""

import numpy as np
import pandas as pd
from copy import copy
from itertools import groupby
from bias_correction import BiasCorrection

crop_seasons = list(range(1993,2017))
homogeneous_groups = list(range(1,5))
months_of_crop_season = list(range(4,11))
month_conversion = {4:"April", 5:"May", 6:"June", 7:"July", 8:"Aug", 9:"Sep", 10:"Oct"} 


def adjust_temperature_bias(observed, predicted, correction_method="normal_mapping"):
    """Apply bias-adjustment to daily temperature values of hindcasts.
    
    parameters:
     - observed: observed climate data
     - predicted: hindcast data
     - correction_method: bias-correction method to be applied. Can be 'normal_mapping', 'quantile_mapping', 'gamma_mapping', 'modified_quantile'
    
    returns: 
     - results: bias-adjusted hindcasts as dataframe  
    """
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


def fill_missing_dates_with_observations(observations, model):
    """Supplement crop season days before initialization date with weather observations.
    
    params:
     - observations: dataframe with weather observations
     - model: dataframe with hindcasts
     
    returns:
     - result: dataframe with hindcast time series supplemented with observations
    """
    
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


def compute_monthly_climate_features(df):
    """Compute monthly climate features."""
    df = df.reset_index().copy()
    df = prepare_drought_count(df)
    grouped_df = (df
                  .groupby(["model", "init_month", "zone", "year", "month"])
                  .agg(
                    Tmean=("tmean", "mean"),
                    Tmax=("tmax", "mean"),
                    Tmin=("tmin", "mean"),
                    Rain=("rain", "sum"),
                    Htemp=("tmax", lambda x: (x > 32).sum()),
                    Ltemp=("tmin", lambda x: (x < 2).sum()),
                    Hrainfall=("rain", lambda x: (x > 30).sum()),
                    Rainy_days=("rain", lambda x: (x > 0.1).sum()),
                    Drought=("consecutives", "sum"))
                  .reset_index())
    
    grouped_df["month"] = grouped_df["month"].replace(month_conversion) 
    grouped_df = grouped_df.pivot(index=["model", "init_month", "zone", "year"], columns="month")
    grouped_df.columns = [s[0] + "_" + s[1] for s in grouped_df.columns]
    
    return grouped_df


def prepare_drought_count(df):
    """Add column with the number of consecutive days without rainfall."""
    df = df.copy().reset_index(drop=False)
    df["new_Value"] = 0
    df["consecutives"] = 0
    grouped = df.groupby(["model", "init_month", "zone", "year"])
    li = []
    for n, gr in grouped:
        l = []
        for k, g in groupby(gr["rain"]):
            size = sum(1 for _ in g)
            if k <= 0.1 and size >= 1:
                l = l + [1]*size
            else:
                l = l + [0]*size
        temp = pd.Series(l)
        temp.index = gr.index
        gr.loc[:, 'new_Value'] = temp
       
        a = gr.loc[:,'new_Value'] != 0
        gr.loc[:,'consecutives'] = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
        li.append(gr)
        
    result = pd.concat(li, axis=0, ignore_index=False)
    result = result.drop({"new_Value"}, axis=1)
    result["consecutives"] = result["consecutives"].apply(lambda x: multiples_of_ten_or_zero(x))
    return result


def multiples_of_ten_or_zero(x):
    """If a multiple of ten, return it, otherwise 0."""
    if x in list(range(10,101,10)):
        return x/10
    else:
        return 0
    

def create_climatology_features(features, climate):
    """Compute features for the climatology approach."""
    climate = climate.loc[climate["month"] >= 8, ["zone", "year", "month", "tmean", "rain"]].reset_index(drop=True).copy()
    features = features.loc[features["model"] == "WS"].reset_index(drop=True).copy()
    climate.columns = ["zone", "year", "month", "Tmean", "Rain"]
    climate["month"] = climate["month"].replace(month_conversion) 
    climate["model"] = "CLIMATE"
    climate = (climate
               .reset_index(drop=True)
               .pivot(index=["model", "year", "zone"], columns="month"))
    climate.columns = [str(s[0]) + "_" + str(s[1]) for s in climate.columns]
    climate = climate.reset_index()
    climate = pd.concat([climate]*8)
    init_months = np.repeat([4, 5, 6, 7, 8, 9, 10, 11], 4*24)
    climate["init_month"] = init_months
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