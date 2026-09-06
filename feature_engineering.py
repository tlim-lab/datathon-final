"""
Feature engineering for one-hour-ahead PM2.5 forecasting.

Input columns available per row: station, observation_timestamp,
current_PM2_5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd
(wind direction, 16-point compass), WSPM (wind speed). The target,
PM2_5_next_hour, is the PM2.5 reading one hour after the row's timestamp
for the same station.

All features below use only information available at the row's own
timestamp (current and past readings) so they are valid at prediction
time for every row, train or test.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Pollutant and weather columns present in the raw data.
POLLUTANTS = ["current_PM2_5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# Standard 16-point compass bearing, in degrees, used to decompose wind
# direction + speed into u/v vector components.
WD_ORDER = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
WD_ANGLE = {d: i * 22.5 for i, d in enumerate(WD_ORDER)}

PM_LAGS = [1, 2, 3, 4, 5, 6, 12, 24]
OTHER_LAGS = [1, 2, 3]
ROLL_WINDOWS = [3, 6, 12, 24]

TOP_K_DONOR_STATIONS = 3


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag, rolling-window, wind, calendar and cross-station
    features for a combined train+test frame.

    A complete hourly timestamp grid is built per station first, so
    that "1 hour ago" always means exactly 1 hour ago even when the
    raw data has gaps (a station with missing readings would otherwise
    make lag_1 silently mean "the previous available row", which is
    not the same thing).
    """
    orig_index = df.index
    df = df.sort_values(["station", "observation_timestamp"]).copy()

    grid = (df.groupby("station").observation_timestamp.agg(["min", "max"])
              .apply(lambda r: pd.date_range(r["min"], r["max"], freq="h"), axis=1)
              .explode().rename("observation_timestamp").reset_index())
    g = grid.merge(df, on=["station", "observation_timestamp"], how="left")
    g = g.sort_values(["station", "observation_timestamp"]).reset_index(drop=True)
    gb = g.groupby("station", sort=False)

    feats = pd.DataFrame(index=g.index)
    feats["station"] = g.station
    feats["observation_timestamp"] = g.observation_timestamp

    for c in POLLUTANTS + WEATHER:
        feats[c] = g[c]
    feats["pm_missing"] = g.current_PM2_5.isna().astype(int)

    # Short sensor gaps (up to 24h) are forward-filled before computing
    # lag/rolling features, so a brief missing reading does not create
    # a chain of NaN lag features for the following day.
    ff = {c: gb[c].ffill(limit=24) for c in POLLUTANTS + WEATHER}
    pm = ff["current_PM2_5"]
    pm_gb = pm.groupby(g.station, sort=False)

    for k in PM_LAGS:
        feats[f"pm_lag{k}"] = pm_gb.shift(k)
    for k in [1, 2, 3, 6]:
        feats[f"pm_diff{k}"] = pm - pm_gb.shift(k)
    feats["pm_accel"] = feats["pm_diff1"] - (pm_gb.shift(1) - pm_gb.shift(2))

    for w in ROLL_WINDOWS:
        r = pm_gb.rolling(w, min_periods=1)
        feats[f"pm_rmean{w}"] = r.mean().reset_index(level=0, drop=True)
        feats[f"pm_rstd{w}"] = r.std().reset_index(level=0, drop=True)
        feats[f"pm_rmax{w}"] = r.max().reset_index(level=0, drop=True)
        feats[f"pm_minus_rmean{w}"] = pm - feats[f"pm_rmean{w}"]
    feats["pm_same_hour_yday"] = pm_gb.shift(24)
    feats["pm_rmean24_diff"] = feats["pm_rmean24"] - feats["pm_rmean24"].groupby(g.station, sort=False).shift(24)

    for c in ["PM10", "NO2", "CO", "O3", "SO2", "TEMP", "PRES", "DEWP", "WSPM"]:
        s = ff[c]
        s_gb = s.groupby(g.station, sort=False)
        for k in OTHER_LAGS:
            feats[f"{c}_lag{k}"] = s_gb.shift(k)
        feats[f"{c}_diff1"] = s - s_gb.shift(1)
        feats[f"{c}_rmean6"] = s_gb.rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)

    feats["pm10_ratio"] = g.current_PM2_5 / (g.PM10 + 1)
    feats["rain_last6"] = ff["RAIN"].groupby(g.station, sort=False).rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    feats["temp_dew_spread"] = g.TEMP - g.DEWP  # a standard humidity proxy

    # Wind direction + speed are decomposed into orthogonal u/v vector
    # components, which is a more natural representation for a model
    # than a raw compass label, since it makes "opposite direction"
    # and "similar direction" a matter of vector distance.
    ang = np.deg2rad(g.wd.map(WD_ANGLE))
    feats["wd_sin"] = np.sin(ang)
    feats["wd_cos"] = np.cos(ang)
    feats["wind_u"] = -g.WSPM * np.sin(ang)
    feats["wind_v"] = -g.WSPM * np.cos(ang)
    feats["wind_u_lag1"] = feats.wind_u.groupby(g.station, sort=False).shift(1)
    feats["wind_v_lag1"] = feats.wind_v.groupby(g.station, sort=False).shift(1)
    feats["wd_code"] = g.wd.map({d: i for i, d in enumerate(WD_ORDER)})

    ts = g.observation_timestamp
    feats["hour"] = ts.dt.hour
    feats["dow"] = ts.dt.dayofweek
    feats["month"] = ts.dt.month
    feats["doy"] = ts.dt.dayofyear
    # Cyclical (sin/cos) encodings avoid the false discontinuity a raw
    # hour/day-of-year value would create between hour 23 and hour 0,
    # or between Dec 31 and Jan 1.
    feats["hour_sin"] = np.sin(2 * np.pi * feats.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * feats.hour / 24)
    feats["doy_sin"] = np.sin(2 * np.pi * feats.doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * feats.doy / 365.25)
    feats["is_weekend"] = (feats.dow >= 5).astype(int)

    # City-wide aggregate: an unweighted snapshot of pollution across
    # all stations at the same hour, useful for city-scale events that
    # affect every station together (e.g. a stagnant-air episode).
    city = pm.groupby(g.observation_timestamp)
    feats["city_mean_pm"] = city.transform("mean")
    feats["city_std_pm"] = city.transform("std")
    feats["city_max_pm"] = city.transform("max")
    feats["pm_minus_city"] = pm - feats.city_mean_pm
    feats["city_mean_lag1"] = feats.city_mean_pm.groupby(g.station, sort=False).shift(1)
    feats["city_diff1"] = feats.city_mean_pm - feats.city_mean_lag1
    feats["city_mean_pm10"] = ff["PM10"].groupby(g.observation_timestamp).transform("mean")
    feats["city_mean_wspm"] = ff["WSPM"].groupby(g.observation_timestamp).transform("mean")

    feats["station"] = feats.station.astype("category")

    key = df[["station", "observation_timestamp"]].reset_index()
    out = key.merge(feats, on=["station", "observation_timestamp"], how="left").set_index("index")
    out = out.reindex(orig_index)
    out["station"] = pd.Categorical(out.station, categories=sorted(df.station.unique()))
    return out


def learn_donor_stations(train: pd.DataFrame, stations, top_k=TOP_K_DONOR_STATIONS):
    """Learn, from the training data only, which other stations'
    current PM2.5 is historically most predictive of each station's
    next-hour PM2.5.

    Air pollution is transported by wind, so nearby or wind-connected
    stations tend to move together. Without station coordinates, this
    correlation-based ranking is a data-driven proxy for that spatial
    relationship: two stations that consistently rise and fall together
    are very likely physically close or frequently wind-linked.
    """
    pivot_cur = train.pivot_table(index="observation_timestamp", columns="station", values="current_PM2_5")
    pivot_next = train.pivot_table(index="observation_timestamp", columns="station", values="PM2_5_next_hour")

    donor_weights = {}
    for s in stations:
        corrs = []
        for other in stations:
            if other == s:
                continue
            pair = pd.concat([pivot_cur[other], pivot_next[s]], axis=1).dropna()
            if len(pair) < 100:
                continue
            c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            corrs.append((other, c))
        corrs.sort(key=lambda x: -x[1])
        donor_weights[s] = corrs[:top_k]
    return donor_weights


def add_spatial_features(F: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, stations) -> pd.DataFrame:
    """Add a cross-station feature: a correlation-weighted average of
    each station's most predictive "donor" stations, at the same hour.

    This uses only same-timestamp readings, which are already known at
    prediction time for every row (the same principle already used by
    city_mean_pm above), so it introduces no future information.
    """
    donor_weights = learn_donor_stations(train, stations)

    grid_cur = (pd.concat([train, test])
                  .pivot_table(index="observation_timestamp", columns="station", values="current_PM2_5")
                  .sort_index().ffill(limit=24))

    rows = []
    for s in stations:
        donors = donor_weights[s]
        weights = np.array([max(c, 0) for _, c in donors])
        if weights.sum() == 0:
            weights = np.ones(len(donors))
        weights = weights / weights.sum()
        vals = np.zeros(len(grid_cur))
        for (donor, _), w in zip(donors, weights):
            vals += grid_cur[donor].fillna(grid_cur[donor].median()).values * w
        rows.append(pd.DataFrame({
            "station": s,
            "observation_timestamp": grid_cur.index,
            "spatial_donor_pm": vals,
        }))
    long_spatial = pd.concat(rows, ignore_index=True)
    long_spatial["station"] = long_spatial["station"].astype(str)

    F = F.copy()
    F["_station_str"] = F["station"].astype(str)
    F = F.merge(long_spatial, left_on=["_station_str", "observation_timestamp"],
                right_on=["station", "observation_timestamp"], how="left", suffixes=("", "_donor"))
    F = F.drop(columns=["_station_str", "station_donor"])
    F["spatial_donor_minus_current"] = F["spatial_donor_pm"] - F["current_PM2_5"]
    return F


def build_full_feature_set(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Build the complete feature table for train+test combined."""
    stations = sorted(train.station.unique())
    all_rows = pd.concat([train.assign(_split="train"), test.assign(_split="test")], ignore_index=True)

    F = build_features(all_rows)
    F["_split"] = all_rows["_split"]
    F["id"] = all_rows["id"]
    F[target_col] = all_rows[target_col]
    F["station"] = F["station"].astype("category")

    F = add_spatial_features(F, train, test, stations)
    F["station"] = F["station"].astype("category")
    return F
