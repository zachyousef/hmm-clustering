#!/usr/bin/env python3
"""
make_hmm_cluster_report.py
==========================
Create a multi‑page PDF for an HMM clustering run.

Page 1  : the 4 metric plots + cluster‑size bar
Pages 2+: for each cluster, the *three patients closest to the cluster
          centroid* (Euclidean distance in 6‑D HMM‑feature space) with:
              • actogram (left)
              • 24‑D distribution (right)

Usage:
    python hmm_report.py  output_plots/hmm_YYYYMMDD_HHMMSS
"""

from __future__ import annotations
import sys, math, random
from pathlib import Path
from datetime import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from hmmlearn.hmm import CategoricalHMM

# ─── paths --------------------------------------------------------
CACHE_RAW   = Path("cache/raw_parquet")
PAT_PNG_DIR = Path("plots/patients")

# ─── helpers (reuse from stage‑1) --------------------------------
def to_time(val):
    if pd.isna(val): return None
    if isinstance(val, time): return val
    try: return pd.to_datetime(val).time()
    except: return None

def t2h(t): return t.hour + t.minute/60 + t.second/3600

def _mark(row, s, e):
    for h in range(int(math.floor(s)), int(math.floor(e))+1):
        if 0<=h<24: row[h] = 1

def load_day_bin(pid: str) -> np.ndarray:
    pq = CACHE_RAW / f"{pid}.parquet"
    df = pd.read_parquet(pq)
    n_rows, n_days = df.shape
    day_bin = np.zeros((n_days,24), dtype=np.uint8)
    for r in range(0, n_rows, 2):
        if r+1 >= n_rows: break
        on_row, off_row = df.iloc[r], df.iloc[r+1]
        for d in range(n_days):
            on, off = to_time(on_row.iloc[d]), to_time(off_row.iloc[d])
            if on is None or off is None: continue
            s,e = t2h(on), t2h(off)
            if e < s:
                _mark(day_bin[d], s, 24)
                if d+1 < n_days: _mark(day_bin[d+1], 0, e)
            else:
                _mark(day_bin[d], s, e)
    return day_bin

def hmm_features(seq: np.ndarray, seed=7) -> np.ndarray:
    if seq.ndim == 1:
        seq = seq.reshape(-1,1)
    model = CategoricalHMM(
        n_components=2,
        n_features=2,
        random_state=seed,
        n_iter=100,
        tol=1e-4,
        init_params="",
        params="ste",
    )
    model.startprob_    = np.array([0.5,0.5])
    model.transmat_     = np.array([[0.9,0.1],[0.1,0.9]])
    model.emissionprob_ = np.array([[1.,0.],[0.,1.]])
    model.fit(seq)

    A = model.transmat_.ravel()
    B = model.emissionprob_[:,1]
    return np.concatenate([A,B]).astype(np.float32)   # 6‑D

def img(p: Path) -> Image.Image: return Image.open(p)
def add_img(ax, im): ax.imshow(im); ax.axis("off")

# ─── main ---------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("usage: python make_hmm_cluster_report.py  <hmm_run_folder>")
        sys.exit(1)

    run = Path(sys.argv[1])
    if not run.exists():
        sys.exit(f"{run} not found")

    labels = np.load(run/"cluster_labels.npy")
    ids    = [ln.strip() for ln in open(run/"patient_ids.txt")]
    k      = len(np.unique(labels))

    # ---------- rebuild HMM feature matrix ----------------------
    feats = []
    print("Recomputing HMM features for centroid distance …")
    for pid in tqdm(ids):
        seq = load_day_bin(pid).flatten()
        feats.append(hmm_features(seq))
    X = np.vstack(feats)

    # ---------- centroid & closest‑3 per cluster ---------------
    closest = {}
    for cl in np.unique(labels):
        idx = np.where(labels==cl)[0]
        centroid = X[idx].mean(axis=0)
        dists = np.linalg.norm(X[idx] - centroid, axis=1)
        order = idx[np.argsort(dists)][:3]    # global indices of top‑3
        closest[cl] = order

    # ---------- open PDF ---------------------------------------
    pdf_path = run / "HMM_Cluster_Report.pdf"
    with PdfPages(pdf_path) as pdf:

        # page 1 – metric overview
        grids = ["elbow_WCSS.png", "silhouette.png",
                 "calinski_harabasz.png", "davies_bouldin.png",
                 "cluster_sizes.png"]
        fig, axes = plt.subplots(3,2,figsize=(8.3,11.7))
        axes = axes.flatten()
        for ax,g in zip(axes,grids):
            add_img(ax, img(run/g))
        for ax in axes[len(grids):]: ax.axis("off")
        fig.suptitle("HMM clustering – metrics", fontsize=14)
        pdf.savefig(fig); plt.close(fig)

        # pages per cluster
        for cl in np.unique(labels):
            idxs = closest[cl]
            n    = len(idxs)
            fig, axarr = plt.subplots(n,2,figsize=(8.3,4*n))
            if n==1: axarr = np.array([axarr])
            for r,gidx in enumerate(idxs):
                pid = ids[gidx]
                add_img(axarr[r,0], img(PAT_PNG_DIR/f"{pid}_actogram.png"))
                add_img(axarr[r,1], img(PAT_PNG_DIR/f"{pid}_distribution.png"))
                axarr[r,0].set_title(f"{pid} – actogram", fontsize=9)
                axarr[r,1].set_title("24‑D distribution", fontsize=9)
            fig.suptitle(f"Cluster {cl}: 3 nearest to centroid", fontsize=14)
            plt.tight_layout(rect=[0,0,1,0.96])
            pdf.savefig(fig); plt.close(fig)

    print("✅  PDF written:", pdf_path)

if __name__ == "__main__":
    main()