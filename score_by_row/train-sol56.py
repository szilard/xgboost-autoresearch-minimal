import pandas as pd
import numpy as np
import time
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold


data_dir = Path(__file__).parent / "data-cache"
train = pd.read_csv(f"{data_dir}/2005-slice1-100k.csv")

cat_cols = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target   = "dep_delayed_15min"


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}
month_start_days = pd.Series(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
    index=range(1, 13),
)
dep_hour_levels = sorted(((train["DepTime"] // 100) % 24).unique())
train_carrier_route = train["UniqueCarrier"] + "-" + train["Origin"] + "-" + train["Dest"]
train_dep_minutes = ((train["DepTime"] // 100) % 24) * 60 + train["DepTime"] % 100
dep_15_minute_levels = sorted((train_dep_minutes // 15).unique())
dep_minute_in_quarter_levels = sorted((train_dep_minutes % 15).unique())
dep_minute_modulo_5_levels = sorted((train_dep_minutes % 5).unique())
carrier_half_hour_levels = sorted(
    (train["UniqueCarrier"] + "-" + (train_dep_minutes // 30).astype(str)).unique()
)
carrier_route_median_dep_minutes = train_dep_minutes.groupby(train_carrier_route).median()
carrier_route_min_dep_minutes = train_dep_minutes.groupby(train_carrier_route).min()
carrier_route_max_dep_minutes = train_dep_minutes.groupby(train_carrier_route).max()
carrier_route_std_dep_minutes = train_dep_minutes.groupby(train_carrier_route).std()

def prepare(df):
    X = df[num_cols + cat_cols].copy()
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    month = df["Month"].str.removeprefix("c-").astype(int)
    day_of_month = df["DayofMonth"].str.removeprefix("c-").astype(int)
    X["DayOfYear"] = month.map(month_start_days) + day_of_month
    dep_hour = (df["DepTime"] // 100) % 24
    X["DepHour"] = pd.Categorical(
        dep_hour.where(dep_hour.isin(dep_hour_levels)),
        categories=dep_hour_levels,
    )
    dep_minutes = dep_hour * 60 + df["DepTime"] % 100
    X["DepMinute"] = df["DepTime"] % 100
    dep_minute_in_quarter = dep_minutes % 15
    X["DepMinuteInQuarter"] = pd.Categorical(
        dep_minute_in_quarter.where(
            dep_minute_in_quarter.isin(dep_minute_in_quarter_levels)
        ),
        categories=dep_minute_in_quarter_levels,
    )
    dep_minute_modulo_5 = dep_minutes % 5
    X["DepMinuteModulo5"] = pd.Categorical(
        dep_minute_modulo_5.where(dep_minute_modulo_5.isin(dep_minute_modulo_5_levels)),
        categories=dep_minute_modulo_5_levels,
    )
    dep_15_minute = dep_minutes // 15
    X["Dep15Minute"] = pd.Categorical(
        dep_15_minute.where(dep_15_minute.isin(dep_15_minute_levels)),
        categories=dep_15_minute_levels,
    )
    dep_angle = 2 * np.pi * dep_minutes / (24 * 60)
    X["DepTimeSin"] = np.sin(dep_angle)
    X["DepTimeCos"] = np.cos(dep_angle)
    carrier_route = df["UniqueCarrier"] + "-" + df["Origin"] + "-" + df["Dest"]
    carrier_route_time_delta = dep_minutes - carrier_route.map(carrier_route_median_dep_minutes)
    X["DepMinutesVsCarrierRouteMedian"] = carrier_route_time_delta
    schedule_min = carrier_route.map(carrier_route_min_dep_minutes)
    schedule_span = carrier_route.map(carrier_route_max_dep_minutes) - schedule_min
    X["CarrierRouteScheduleSpan"] = schedule_span
    X["CarrierRouteScheduleStd"] = carrier_route.map(carrier_route_std_dep_minutes)
    X["CarrierRouteSchedulePosition"] = (dep_minutes - schedule_min) / schedule_span.replace(0, np.nan)
    carrier_half_hour = df["UniqueCarrier"] + "-" + (dep_minutes // 30).astype(str)
    X["CarrierHalfHour"] = pd.Categorical(
        carrier_half_hour.where(carrier_half_hour.isin(carrier_half_hour_levels)),
        categories=carrier_half_hour_levels,
    )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = xgb.XGBClassifier(
    n_estimators=900,
    max_depth=9,
    learning_rate=0.11,
    reg_lambda=32.0,
    colsample_bytree=0.7,
    max_cat_threshold=24,
    max_cat_to_onehot=6,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)


model.fit(X_train, y_train)

