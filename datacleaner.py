#!/usr/bin/env python3
"""
datacleaner.py
──────────────
• Reads every .xlsx workbook in DATA_FOLDER (all sheets, header=None).
• Each sheet: first 2 columns are meta / on–off flag, every later column = one day.
• Detects the “real” start day with unsupervised change‑point detection (ruptures‑PELT).
• Drops all day‑columns before that start day.
• Writes an identical workbook tree under OUT_FOLDER.
• Prints a before‑vs‑after usage summary (per‑sheet + dataset roll‑up).

Dependencies
------------
pip install pandas numpy ruptures openpyxl
"""

import os, glob
from datetime import datetime, time

import numpy as np
import pandas as pd
import ruptures as rpt
from openpyxl import Workbook           # just so pandas/openpyxl combo works

# ────────────────────────────────────────────────────────────────────
# 0)  PATHS  – update BASE_DIR if needed
# ────────────────────────────────────────────────────────────────────
BASE_DIR    = "/Users/zachyousef/Documents/RSA/MatlabPython"
DATA_FOLDER = os.path.join(BASE_DIR, "Data")
OUT_FOLDER  = os.path.join(BASE_DIR, "Data_Corrected")

# ────────────────────────────────────────────────────────────────────
# 1)  HELPERS
# ────────────────────────────────────────────────────────────────────
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def hhmm_to_min(val):
    """Convert 'hh:mm[:ss]' or Excel time → minutes past midnight."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (datetime, time)):
        return val.hour * 60 + val.minute + val.second / 60
    s = str(val).strip()
    if ":" in s:
        try:
            parts = list(map(int, s.split(":")))
            h, m = parts[0], parts[1]
            sec = parts[2] if len(parts) == 3 else 0
            return h * 60 + m + sec / 60
        except ValueError:
            pass
    try:                               # Excel fractional day (0–1)
        return float(s) * 24 * 60
    except ValueError:
        return np.nan

def usage_for_column(col: pd.Series) -> float:
    """
    Sum all on/off pairs down a single day‑column. Handles midnight wrap.
    Rows alternate: on / off / on / off …
    """
    vals = col.dropna().reset_index(drop=True)
    if len(vals) < 2:
        return 0.0
    if len(vals) % 2:                  # odd rows → drop dangling 'on'
        vals = vals.iloc[:-1]
    total = 0.0
    for i in range(0, len(vals), 2):
        t_on, t_off = hhmm_to_min(vals[i]), hhmm_to_min(vals[i + 1])
        if np.isnan(t_on) or np.isnan(t_off):
            continue
        diff = t_off - t_on
        if diff < 0:                   # crossed midnight
            diff += 24 * 60
        total += diff
    return total

def detect_real_start(usage_vec, min_jump=30.0):
    """
    Find first change‑point where mean usage jumps ≥ min_jump minutes.
    Fallback: first non‑zero column.
    """
    sig = np.array(usage_vec).reshape(-1, 1)
    if len(sig) < 4:
        return next((i for i, u in enumerate(usage_vec) if u > 0), 0)

    pen = max(10, np.log(len(sig)) * 5)        # mild regularisation
    cp  = rpt.Pelt(model="l2").fit(sig).predict(pen=pen)
    for k in cp[:-1]:                          # skip final len(sig)
        left_mean, right_mean = np.mean(sig[:k]), np.mean(sig[k:])
        if right_mean - left_mean >= min_jump:
            return k
    return next((i for i, u in enumerate(usage_vec) if u > 0), 0)

def trim_sheet(df: pd.DataFrame, first_day_col=2) -> pd.DataFrame:
    """Return a copy with columns < real start removed. No change if <3 cols."""
    if df.shape[1] <= first_day_col:
        return df.copy()

    day_cols = list(range(first_day_col, df.shape[1]))
    nightly  = [usage_for_column(df.iloc[:, c]) for c in day_cols]
    k_start  = detect_real_start(nightly)

    keep = [0, 1] + day_cols[k_start:]
    keep = [c for c in keep if c < df.shape[1]]  # clamp
    return df.iloc[:, keep]

def sheet_diff(original: pd.DataFrame, trimmed: pd.DataFrame,
               first_day_col=2) -> dict:
    """Return simple diff stats for one sheet."""
    if original.shape[1] <= first_day_col:
        return {"removed": 0, "pre_mean": 0.0, "post_mean": 0.0}

    day_cols  = list(range(first_day_col, original.shape[1]))
    nightly   = [usage_for_column(original.iloc[:, c]) for c in day_cols]

    removed   = original.shape[1] - trimmed.shape[1]
    cut_point = removed - (first_day_col - 2)           # num dropped day‑cols
    pre_vec   = nightly[:cut_point]
    post_vec  = nightly[cut_point:]

    return {"removed": cut_point,
            "pre_mean": float(np.mean(pre_vec))  if pre_vec  else 0.0,
            "post_mean": float(np.mean(post_vec)) if post_vec else 0.0}

# ────────────────────────────────────────────────────────────────────
# 2)  MAIN
# ────────────────────────────────────────────────────────────────────
def main():
    ensure_dir(OUT_FOLDER)
    writer_cache = {}          # keep each target workbook open once
    diff_stats   = []          # per‑sheet diffs

    for wb_path in sorted(glob.glob(os.path.join(DATA_FOLDER, "*.xlsx"))):
        try:
            xls = pd.ExcelFile(wb_path)
        except Exception as e:
            print(f"⚠️  Cannot open {wb_path}: {e}")
            continue

        for sh in xls.sheet_names:
            try:
                raw = pd.read_excel(wb_path, sheet_name=sh, header=None,
                                    na_values=["", "NaN"], keep_default_na=False)
            except Exception as e:
                print(f"⚠️  Skip {os.path.basename(wb_path)} / {sh}: {e}")
                continue

            trimmed = trim_sheet(raw)
            diff_stats.append(sheet_diff(raw, trimmed))

            out_path = os.path.join(OUT_FOLDER, os.path.basename(wb_path))
            ensure_dir(os.path.dirname(out_path))
            if out_path not in writer_cache:
                writer_cache[out_path] = pd.ExcelWriter(out_path, engine="openpyxl")
            trimmed.to_excel(writer_cache[out_path], sheet_name=sh,
                             header=False, index=False)

        print("✔", os.path.basename(wb_path), "processed")

    # close all writers
    for w in writer_cache.values():
        w.close()

    # ─────  SUMMARY  ─────
    if diff_stats:
        df_d = pd.DataFrame(diff_stats)
        any_cut = df_d["removed"] > 0
        print("\n─────────  CHANGE SUMMARY  ─────────")
        print(f"Sheets processed          : {len(df_d)}")
        print(f"Sheets trimmed            : {any_cut.sum()} "
              f"({any_cut.mean()*100:.1f} %)")
        if any_cut.any():
            print(f"Avg. columns removed      : {df_d.loc[any_cut,'removed'].mean():.1f}")
        print(f"Mean usage before trim    : {df_d['pre_mean'].mean():.1f} min")
        print(f"Mean usage after  trim    : {df_d['post_mean'].mean():.1f} min")
        print("────────────────────────────────────")
        # Optional: save per‑sheet diff
        # df_d.to_csv(os.path.join(OUT_FOLDER, "trim_summary.csv"), index=False)

    print("\n✓  Corrected workbooks saved in", OUT_FOLDER)

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
