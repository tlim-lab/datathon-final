"""
Train the LightGBM component of the final model.

The target is modeled as a delta (PM2_5_next_hour - current_PM2_5)
rather than the raw next-hour level. PM2.5 is strongly autocorrelated
hour to hour, so the change between consecutive hours is a much more
stationary and easier quantity to model than the absolute level; the
predicted delta is added back to the current reading to recover the
final forecast. Huber loss is used instead of squared error because
PM2.5 has occasional large, sharp spikes (heating-season pollution
episodes) that would otherwise dominate a squared-error objective.

Five models with different random seeds are trained and averaged.
LightGBM's row/feature bagging is stochastic, so different seeds
produce slightly different trees; averaging reduces that variance
without changing what the model has learned overall.

Hyperparameters below were selected via time-based cross-validation:
the training period was split at two points, holding out a later block
of months as a validation window each time, so the fold mimics the
actual situation of predicting a period after the training data ends.
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from feature_engineering import build_full_feature_set

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "PM2_5_next_hour"
N_ROUNDS = 3800
SEEDS = [42, 43, 44, 45, 46]

LGB_PARAMS = {
    "objective": "huber",
    "alpha": 65.6819986135223,
    "metric": "rmse",
    "learning_rate": 0.02110200828255171,
    "num_leaves": 91,
    "min_data_in_leaf": 13,
    "feature_fraction": 0.9993680431002837,
    "bagging_fraction": 0.774779381166933,
    "bagging_freq": 1,
    "lambda_l1": 0.06219885063025651,
    "lambda_l2": 0.1420975136398397,
    "min_gain_to_split": 0.6751376841209982,
    "max_depth": 11,
    "num_threads": os.cpu_count(),
    "verbose": -1,
}


def main():
    t0 = time.time()

    train = pd.read_csv(os.path.join(HERE, "train.csv"), parse_dates=["observation_timestamp"])
    test = pd.read_csv(os.path.join(HERE, "test.csv"), parse_dates=["observation_timestamp"])

    F = build_full_feature_set(train, test, TARGET)
    feature_cols = [c for c in F.columns if c not in {"observation_timestamp", "_split", "id", TARGET}]

    TR = F[F._split == "train"].copy()
    TE = F[F._split == "test"].copy()
    print(f"Feature set ready: {len(feature_cols)} features, {len(TR)} train rows, {len(TE)} test rows  [{time.time()-t0:.1f}s]")

    # If a reading is missing for the current hour, fall back to the
    # last known reading for that station rather than leaving a gap.
    base_tr = TR.current_PM2_5.fillna(TR.pm_lag1).fillna(TR[TARGET].mean())
    base_te = TE.current_PM2_5.fillna(TE.pm_lag1).fillna(TR[TARGET].mean())

    predictions = []
    for seed in SEEDS:
        dtrain = lgb.Dataset(TR[feature_cols], TR[TARGET] - base_tr, categorical_feature=["station"])
        params = {**LGB_PARAMS, "seed": seed, "bagging_seed": seed}
        model = lgb.train(params, dtrain, num_boost_round=N_ROUNDS)
        pred = np.clip(model.predict(TE[feature_cols]) + base_te.values, 0, None)
        predictions.append(pred)
        print(f"  seed {seed} trained  [{time.time()-t0:.1f}s]")

    avg_pred = np.mean(predictions, axis=0)

    out_path = os.path.join(HERE, "lightgbm_predictions.csv")
    pd.DataFrame({"id": TE.id.values, TARGET: avg_pred}).to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
