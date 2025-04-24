"""
===============================================================
              CPAP Usage ‑> Cached ML Pre‑pipeline
===============================================================
* Reads every .xlsx workbook inside  New_Data/
* Extracts every sheet (one sheet == one patient recording)
* Caches
    ─ raw noon‑to‑noon grid   →  Parquet
    ─ 24‑D usage vector       →  .npy
    ─ actogram / distribution →  .png  (only rendered once)
----------------------------------------------------------------
After this script finishes you have, for every patient:

    cache/
        raw_parquet/{patient_id}.parquet
        binned/{patient_id}.npy
    plots/patients/
        {patient_id}_actogram.png
        {patient_id}_distribution.png

All subsequent ML work can load the tiny 24‑D vectors in milliseconds.
"""

from __future__ import annotations
from pathlib import Path
from datetime import time
import math, hashlib, json, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────── config ──────────────────────────────
NEW_DATA_DIR      = Path("Data_Corrected")
CACHE_RAW_DIR     = Path("cache/raw_parquet")
CACHE_BINNED_DIR  = Path("cache/binned")
PLOTS_DIR         = Path("plots/patients")
CACHE_RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_BINNED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────── helper: time/interval utils ─────────────────
def to_time(val) -> time | None:
    """Coerce spreadsheet cell to `datetime.time` (or None)."""
    if pd.isna(val):
        return None
    if isinstance(val, time):
        return val
    try:
        return pd.to_datetime(val).time()
    except (ValueError, TypeError):
        return None

def t2h(t: time) -> float:
    return t.hour + t.minute / 60 + t.second / 3600

# ───────────────────── raw ingest  (step 1) ────────────────────────
def read_or_load_parquet(xlsx: Path, sheet_name) -> pd.DataFrame:
    """
    Return a noon‑to‑noon grid (rows ≈ 30, columns ≈ 988) for one patient.
    Cached on disk as Parquet keyed by workbook name + sheet.
    """
    pid = f"{xlsx.stem}_sheet{sheet_name}"
    pq_path = CACHE_RAW_DIR / f"{pid}.parquet"

    # regenerate cache if parquet missing or older than source .xlsx
    if not pq_path.exists() or pq_path.stat().st_mtime < xlsx.stat().st_mtime:
        print(f"  • parsing  {pid}  from Excel …")
        df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
        df.to_parquet(pq_path, engine="pyarrow", index=False)
    else:
        df = pd.read_parquet(pq_path, engine="pyarrow")

    return df

# ───────────────────── binned usage  (step 2) ──────────────────────
def compute_or_load_vector(pid: str, df: pd.DataFrame) -> np.ndarray:
    """
    Builds the 24‑D representative vector using the *old* rules:
        • inclusive off‑hour (via _mark_full_hour)
        • divide by total number of sheet‑days (even zero‑usage days)
    """
    vec_path = CACHE_BINNED_DIR / f"{pid}.npy"
    pq_mtime  = (CACHE_RAW_DIR / f"{pid}.parquet").stat().st_mtime
    if vec_path.exists() and vec_path.stat().st_mtime > pq_mtime:
        return np.load(vec_path)

    n_rows, n_days = df.shape
    day_bin = np.zeros((n_days, 24), dtype=np.uint8)

    for r in range(0, n_rows, 2):
        if r + 1 >= n_rows:
            break
        on_row, off_row = df.iloc[r], df.iloc[r + 1]
        for d in range(n_days):
            on_t, off_t = to_time(on_row[d]), to_time(off_row[d])
            if on_t is None or off_t is None:
                continue
            s, e = t2h(on_t), t2h(off_t)
            if e < s:                                    # crosses midnight
                _mark_full_hour(day_bin[d],     s, 24)
                if d + 1 < n_days:
                    _mark_full_hour(day_bin[d+1], 0, e)
            else:
                _mark_full_hour(day_bin[d], s, e)

    # OLD behaviour: divide by *all* days (including zeros)
    vector = day_bin.mean(axis=0).astype(np.float32)
    np.save(vec_path, vector)
    return vector

def _mark_full_hour(binary_row: np.ndarray, start: float, end: float) -> None:
    """
    Legacy binning: include *entire* floor(start)..floor(end) hours, regardless
    of the exact minute the device turned off.
    Example 03:12‑04:05 marks hour bins 3 *and* 4.
    """
    s_hr = int(math.floor(start))
    e_hr = int(math.floor(end))
    for h in range(s_hr, e_hr + 1):          # ‹= inclusive›
        if 0 <= h < 24:
            binary_row[h] = 1

# ──────────────────── patient‑level plots  (step 3) ────────────────
def cache_plots(pid: str, df: pd.DataFrame, vec: np.ndarray) -> None:
    act_path = PLOTS_DIR / f"{pid}_actogram.png"
    dist_path = PLOTS_DIR / f"{pid}_distribution.png"

    if not act_path.exists():
        _plot_actogram(df, act_path)
    if not dist_path.exists():
        _plot_distribution(vec, dist_path)

def _plot_actogram(df: pd.DataFrame, out: Path,
                   row_height: float = 0.07,          # ↓ from 0.16
                   tick_every: int | None = None):    # new
    n_rows, n_days = df.shape
    fig_h = max(4, row_height * n_days)              # 1️⃣  tighter figure
    fig, ax = plt.subplots(figsize=(10, fig_h))

    for d in range(n_days):
        y = d + 1
        for r in range(0, n_rows, 2):
            if r + 1 >= n_rows:
                break
            on_t, off_t = to_time(df.iat[r, d]), to_time(df.iat[r + 1, d])
            if on_t is None or off_t is None:
                continue
            s, e = t2h(on_t), t2h(off_t)
            if e < s:                      # crosses midnight
                ax.hlines(y, s, 24, lw=.7)
                ax.hlines(y + 1, 0, e, lw=.7)
            else:
                ax.hlines(y, s, e, lw=.7)

    # ───── y‑axis formatting tweaks ──────────────────────────────────
    if tick_every is None:
        # default: ~10 ticks max
        tick_every = max(1, n_days // 10)
    visible_days = range(1, n_days + 1, tick_every)

    ax.set_ylim(0.5, n_days + 0.5)
    ax.set_yticks(visible_days)                      # 2️⃣  sparse ticks
    ax.set_yticklabels(visible_days)

    # x‑axis unchanged
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 7, 19, 24])
    ax.set_xticklabels(['00:00', '07:00', '19:00', '00:00'])
    ax.invert_yaxis()
    ax.axvspan(7, 19, facecolor='yellow', alpha=.25)
    ax.grid(True, axis='x', linestyle='--', linewidth=.3)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day')
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)

def _plot_distribution(vec: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(24), vec, width=.8)
    ax.set_xticks([0, 7, 19, 23])
    ax.set_xticklabels(['00', '07', '19', '23'])
    ax.set_xlabel('Hour')
    ax.set_ylabel('Fraction of days with usage')
    ax.set_ylim(0, 1)
    ax.set_title('24‑D Binned Usage Distribution')
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)

# ───────────────────────────── driver ──────────────────────────────
def main():
    xlsxs = sorted(NEW_DATA_DIR.glob("*.xlsx"))
    if not xlsxs:
        sys.exit("No .xlsx files found in New_Data/")

    for wb in xlsxs:
        xl = pd.ExcelFile(wb)
        print(f"\n=== Workbook: {wb.name} ({len(xl.sheet_names)} sheets) ===")
        for sheet in xl.sheet_names:
            pid = f"{wb.stem}_sheet{sheet}"
            df = read_or_load_parquet(wb, sheet)
            vec = compute_or_load_vector(pid, df)
            cache_plots(pid, df, vec)
            print(f"    Cached: {pid}")

if __name__ == "__main__":
    main()