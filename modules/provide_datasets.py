#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provide_datasets contains functions to read files and prepate datasets.

This data includes:
 - retrospective seasonal climate forecasts (hindcasts) from ECMWF, UKMO, and NCEP
 - weather observations from four locations in Brazil
 - historical wheat yield, both on group and national level
 - climatological data that is computed from climate observations
"""

import glob
import pandas as pd

crop_seasons = list(range(1993,2017))
months_of_crop_season = list(range(4,11))
month_conversion = {4:"April", 5:"May", 6:"June", 7:"July", 8:"Aug", 9:"Sep", 10:"Oct"} 


def read_raw_model_data():
    """Read raw hindcast data.
    
    returns:
     - hindcasts: dataframe with daily raw output of seasonal climate models from ECMWF, UKMO, NCEP, and their averaged output MME
    """ 
    ukmo = pd.read_csv("data/Raw Hindcasts as CSV/ukmo.csv", dtype={"date":str, "group":int}, parse_dates=["date"])
    ncep = pd.read_csv("data/Raw Hindcasts as CSV/ncep.csv", dtype={"date":str, "group":int}, parse_dates=["date"])
    ecmwf = pd.read_csv("data/Raw Hindcasts as CSV/ecmwf.csv", dtype={"date":str, "group":int}, parse_dates=["date"])
    df = pd.concat([ukmo, ncep, ecmwf])
    df = df.sort_values(by=["model", "init_month", "ensemble", "group", "year", "month", "date"])
    
    ensemble_aggregation = (df
                            .groupby(["model", "init_month", "group", "year", "month", "date"])
                            .agg({"tmean":"mean", "tmax":"mean", "tmin":"mean", "rain":"mean"})
                            .reset_index(drop=False))

    multi_model_ensemble = (ensemble_aggregation
                            .groupby(["init_month", "group", "year", "month", "date"])
                            .agg({"tmean":"mean", "tmax":"mean", "tmin":"mean", "rain":"mean"})
                            .reset_index(drop=False)
                            .assign(model="MME")
                            .loc[:, ensemble_aggregation.columns])

    hindcasts = (pd.concat([ensemble_aggregation, multi_model_ensemble])
                 .rename(columns={"date":"time", "group":"zone"})
                 .set_index(["model", "init_month", "zone", "year", "month"])
                 .sort_index())
    
    return hindcasts


def read_observed_weather():
    """Read daily weather observations for each group and return it in one dataframe.
    
    returns:
     - observations: dataframe with daily climate observations of tmean, tmax, tmin, and rain
    """
    weather_station_to_group_id = {"PFUN":1, "LOND":2, "CAMP":3, "PGRO":4}         
    all_files = glob.glob("data/Observed Weather/*.csv")
    li = []
    for _, filename in enumerate(all_files):
        observations = pd.read_csv(filename,
                                   usecols=["date", "rain", "tmax", "tmin", "tmean", "treatment"], 
                                   dtype={"date":str}, 
                                   parse_dates=["date"])
        li.append(observations)
        
    observations = (pd.concat(li, axis=0, ignore_index=False)
                    .assign(
                        model="WS", 
                        init_month=11, 
                        year=lambda x: x["date"].dt.year, 
                        month=lambda x: x["date"].dt.month)
                   )
    observations = (observations.loc[(observations["month"].isin(months_of_crop_season))
                         & (observations["year"].isin(crop_seasons))].reset_index(drop=True))
    
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
    """Calculate average climate conditions for each year, location, and month using all other years.
    
    params:
     - observed: dataframe with climate observations to be used to calculate average climate
     
    returns:
     - climatology: dataframe with average climate for each year, location, and month 
    """
    observed = observed.copy()
    li = []
    for year in crop_seasons:
        years_for_climatology = [y for y in crop_seasons if y != year]
        climatology = (observed
                       .loc[("WS", 11, [1, 2, 3, 4], years_for_climatology, months_of_crop_season)]
                       .groupby(["zone", "year", "month"])
                       .agg({"tmean":"mean", "tmax":"mean", "tmin":"mean", "rain":"sum"})
                       .reset_index()
                       .groupby(["zone", "month"])
                       .agg({"tmean":"mean", "tmax":"mean", "tmin":"mean", "rain":"mean"})
                       .reset_index()
                       .assign(year=year)
                       .copy())
        climatology.insert(1, 'year', climatology.pop('year'))
        li.append(climatology)
    climatology = pd.concat(li, axis=0, ignore_index=False)
    return climatology


def read_wheat_yield_data():
    """Read and return detrended wheat yield on national and group level.
    
    returns:
     - national_yield: dataframe with detrended national wheat yield from 1993 to 2016
     - yield_by_group: dataframe with detrended wheat yield by group from 1993 to 2016
    """
    national_yield = pd.read_csv("data/Wheat/ibge_national_yield_detrended.csv")
    yield_by_group = pd.read_csv("data/Wheat/yield_by_group_detrended.csv")
    return (national_yield, yield_by_group)