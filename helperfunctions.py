"""Libraries"""
import glob
import numpy as np
import pandas as pd
from copy import copy
from itertools import groupby
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from bias_correction import BiasCorrection

"""Variables"""
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

# Read Data
def read_raw_model_data():
    """Reads and returns raw hindcasts from ECMWF, UKMO, NCEP and MME as one dataframe."""
    
    ukmo = pd.read_csv("Data/Raw Hindcasts as CSV/ukmo.csv", dtype={"date":str, "group":int}, parse_dates=["date"])
    ncep = pd.read_csv("Data/Raw Hindcasts as CSV/ncep.csv", dtype={"date":str, "group":int}, parse_dates=["date"])
    ecmwf = pd.read_csv("Data/Raw Hindcasts as CSV/ecmwf.csv", dtype={"date":str, "group":int}, parse_dates=["date"])
    
    df = pd.concat([ukmo, ncep, ecmwf])
    df = df.sort_values(by=["model", "init_month", "ensemble", "group", "year", "month", "date"])
    df = df.loc[(df["init_month"] >= 4) & (df["month"] >= 4)].reset_index(drop=True)
    
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
    observations = (observations
                    .loc[(observations["month"].isin(months_of_crop_season))
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

# Bias Adjustment
def adjust_temperature_bias(observed, predicted, correction_method="normal_mapping"):
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


def compute_monthly_climate_features(df):
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
    """Adds a column with the number of consecutive days without rainfall"""
    
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
    if x in list(range(10,101,10)):
        return x/10
    else:
        return 0
    

def create_climatology_features(features, climate):
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


# Yield Data
def read_wheat_yield_data():
    national_yield = pd.read_csv("Data/Wheat/ibge_national_yield_detrended.csv")
    yield_by_group = pd.read_csv("Data/Wheat/yield_by_group_detrended.csv")
    return (national_yield, yield_by_group)



## 1st experiment ##
def retrain_weights(data, yield_df, model="ECMWF", init=8):
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

# 2nd, 3rd, 4th experiment
def calculate_estimates(data, yield_df, model="ECMWF", init=8, no_of_features=6):
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
