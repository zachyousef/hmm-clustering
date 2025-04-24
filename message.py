# hist_adherence_and_days.py
import math
from pathlib import Path
from datetime import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

RAW_DIR = Path("cache/raw_parquet")

# ── helpers (copied from stage‑1) ───────────────────────────────
def to_time(val):
    if pd.isna(val): return None
    if isinstance(val, time): return val
    try: return pd.to_datetime(val).time()
    except: return None

def t2h(t): return t.hour + t.minute/60 + t.second/3600
def _mark(row, s, e):
    for h in range(int(math.floor(s)), int(math.floor(e))+1):
        if 0 <= h < 24:
            row[h] = 1

def load_day_bin(pq: Path):
    df = pd.read_parquet(pq)
    n_rows, n_days = df.shape
    day_bin = np.zeros((n_days, 24), dtype=np.uint8)
    for r in range(0, n_rows, 2):
        if r + 1 >= n_rows: break
        on_row, off_row = df.iloc[r], df.iloc[r+1]
        for d in range(n_days):
            on, off = to_time(on_row.iloc[d]), to_time(off_row.iloc[d])
            if on is None or off is None: continue
            s, e = t2h(on), t2h(off)
            if e < s:
                _mark(day_bin[d],     s, 24)
                if d + 1 < n_days:
                    _mark(day_bin[d+1], 0, e)
            else:
                _mark(day_bin[d], s, e)
    return day_bin
# ────────────────────────────────────────────────────────────────

adherences = []
total_days = []

parquets = sorted(RAW_DIR.glob("*.parquet"))
for pq in tqdm(parquets, desc="Calculating adherence & day counts"):
    db = load_day_bin(pq)              # binary matrix   days × 24
    tot  = db.shape[0]
    act  = (db.sum(axis=1) > 0).sum()
    total_days.append(tot)
    adherences.append(act / tot if tot else np.nan)

adherences = np.array(adherences, dtype=float)
total_days = np.array(total_days, dtype=int)

print(f"Patients processed  : {len(adherences)}")
print(f"Mean adherence      : {np.nanmean(adherences):.3f}")
print(f"Median adherence    : {np.nanmedian(adherences):.3f}")
print(f"Mean days recorded  : {total_days.mean():.1f}")
print(f"Median days recorded: {np.median(total_days):.0f}")

# ── twin histograms ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(adherences[~np.isnan(adherences)],
             bins=20, color="skyblue", edgecolor="black")
axes[0].set(xlabel="Adherence fraction",
            ylabel="Number of patients",
            title="Histogram of adherence")

axes[1].hist(total_days,
             bins=min(30, len(np.unique(total_days))),
             color="lightgreen", edgecolor="black")
axes[1].set(xlabel="Total days in record",
            title="Histogram of days per patient")

plt.tight_layout()
plt.show()