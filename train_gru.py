"""
Train the GRU component of the final model.

This is a recurrent sequence model: instead of hand-built lag and
rolling-window features, it reads the raw last 24 hours of readings for
a station directly and lets the network learn its own representation
of recent history. It is trained on the same delta target as the
LightGBM model (predicted change from the current reading), using a
learned per-station embedding so the network can pick up on
station-specific baseline behaviour.

This script is kept in its own process, separate from the LightGBM
script. LightGBM and PyTorch each initialize their own multi-threaded
math library, and loading both in the same Python process can cause
LightGBM's thread pool to hang on some systems; running them as
separate processes avoids that entirely.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from feature_engineering import build_features, WD_ANGLE

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "PM2_5_next_hour"

SEED = 42
SEQ_LEN = 24
HIDDEN_SIZE = 64
EMBEDDING_SIZE = 4
EPOCHS = 8
BATCH_SIZE = 1024
LEARNING_RATE = 2e-3

# Raw per-hour signals the sequence model reads directly, rather than
# through hand-built lag/rolling features.
SEQUENCE_COLUMNS = [
    "current_PM2_5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "wind_u", "wind_v", "hour_sin", "hour_cos", "city_mean_pm",
]


class GRUForecaster(nn.Module):
    def __init__(self, n_features, n_stations, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.station_embedding = nn.Embedding(n_stations, embedding_size)
        self.gru = nn.GRU(n_features + embedding_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_features + embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, station_ids):
        emb = self.station_embedding(station_ids)[:, None, :].expand(-1, x.shape[1], -1)
        z = torch.cat([x, emb], dim=-1)
        hidden, _ = self.gru(z)
        last_step = torch.cat([hidden[:, -1], z[:, -1]], dim=-1)
        return self.head(last_step).squeeze(-1)


def main():
    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train = pd.read_csv(os.path.join(HERE, "train.csv"), parse_dates=["observation_timestamp"])
    test = pd.read_csv(os.path.join(HERE, "test.csv"), parse_dates=["observation_timestamp"])
    stations = sorted(train.station.unique())

    all_rows = pd.concat([train.assign(_split="train"), test.assign(_split="test")], ignore_index=True)
    F = build_features(all_rows)
    F["_split"] = all_rows["_split"]
    F["id"] = all_rows["id"]
    F[TARGET] = all_rows[TARGET]
    F["station"] = F["station"].astype("category")

    TR = F[F._split == "train"].copy()
    TE = F[F._split == "test"].copy()
    base_tr = TR.current_PM2_5.fillna(TR.pm_lag1).fillna(TR[TARGET].mean())
    base_te = TE.current_PM2_5.fillna(TE.pm_lag1).fillna(TR[TARGET].mean())

    # A complete hourly grid across train+test, indexed by station and
    # timestamp, is used to look up each row's preceding 24-hour window.
    grid = (pd.concat([train, test]).sort_values(["station", "observation_timestamp"])
              .set_index(["station", "observation_timestamp"]))
    full_index = pd.MultiIndex.from_tuples(
        [(s, ts) for s in stations
                 for ts in pd.date_range(train.observation_timestamp.min(), test.observation_timestamp.max(), freq="h")],
        names=["station", "observation_timestamp"])
    grid = grid.reindex(full_index)
    grid["wind_u"] = -grid.WSPM * np.sin(np.deg2rad(grid.wd.map(WD_ANGLE)))
    grid["wind_v"] = -grid.WSPM * np.cos(np.deg2rad(grid.wd.map(WD_ANGLE)))
    grid["hour_sin"] = np.sin(2 * np.pi * grid.index.get_level_values(1).hour / 24)
    grid["hour_cos"] = np.cos(2 * np.pi * grid.index.get_level_values(1).hour / 24)
    grid["city_mean_pm"] = grid.groupby(level=1).current_PM2_5.transform("mean")

    seq_data = grid[SEQUENCE_COLUMNS].copy()
    missing_flags = seq_data[["current_PM2_5", "PM10", "NO2", "CO", "O3"]].isna().astype(np.float32).add_suffix("_missing")
    seq_data = seq_data.groupby(level=0).ffill(limit=24)
    seq_data = seq_data.fillna(seq_data.median())
    seq_data["doy_sin"] = np.sin(2 * np.pi * grid.index.get_level_values(1).dayofyear / 365.25)
    seq_data["doy_cos"] = np.cos(2 * np.pi * grid.index.get_level_values(1).dayofyear / 365.25)
    seq_data = pd.concat([seq_data, missing_flags], axis=1)

    train_end = train.observation_timestamp.max()
    mean = seq_data[seq_data.index.get_level_values(1) <= train_end].mean()
    std = seq_data[seq_data.index.get_level_values(1) <= train_end].std() + 1e-6
    seq_tensor = torch.tensor(((seq_data - mean) / std).values, dtype=torch.float32).to(device)

    grid_position = pd.Series(np.arange(len(seq_data)), index=seq_data.index)
    station_ids_map = {s: i for i, s in enumerate(stations)}
    pm_scale = float(train.current_PM2_5.std())

    def lookup_positions(index, ref_df):
        station_names = ref_df.loc[index, "station"].astype(str)
        keys = list(zip(station_names, ref_df.loc[index, "observation_timestamp"]))
        positions = torch.tensor(grid_position.loc[keys].values)
        station_ids = torch.tensor(station_names.map(station_ids_map).values)
        return positions, station_ids

    torch.manual_seed(SEED)
    offsets = torch.arange(-SEQ_LEN + 1, 1).to(device)

    train_positions, train_station_ids = lookup_positions(TR.index, F)
    test_positions, test_station_ids = lookup_positions(TE.index, F)
    train_positions, train_station_ids = train_positions.to(device), train_station_ids.to(device)
    test_positions, test_station_ids = test_positions.to(device), test_station_ids.to(device)

    train_targets = torch.tensor((TR[TARGET].values - base_tr.values) / pm_scale, dtype=torch.float32).to(device)
    valid_rows = train_positions >= SEQ_LEN - 1
    train_positions, train_station_ids, train_targets = train_positions[valid_rows], train_station_ids[valid_rows], train_targets[valid_rows]

    model = GRUForecaster(seq_tensor.shape[1], len(station_ids_map)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    steps_per_epoch = (len(train_positions) + BATCH_SIZE - 1) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=EPOCHS * steps_per_epoch)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_positions))
        for i in range(0, len(perm), BATCH_SIZE):
            batch = perm[i:i + BATCH_SIZE]
            batch_seq = seq_tensor[train_positions[batch, None] + offsets]
            loss = nn.functional.mse_loss(model(batch_seq, train_station_ids[batch]), train_targets[batch])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(f"  epoch {epoch + 1}/{EPOCHS} done  [{time.time()-t0:.1f}s]")

    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(test_positions), 8192):
            batch_seq = seq_tensor[test_positions[i:i + 8192, None] + offsets]
            outputs.append(model(batch_seq, test_station_ids[i:i + 8192]))
    pred = np.clip(torch.cat(outputs).cpu().numpy() * pm_scale + base_te.values, 0, None)

    out_path = os.path.join(HERE, "gru_predictions.csv")
    pd.DataFrame({"id": TE.id.values, TARGET: pred}).to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
