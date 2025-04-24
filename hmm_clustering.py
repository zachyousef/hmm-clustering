#!/usr/bin/env python3
"""
=============================================================
          CPAP  –  HMM-based consensus-ready clustering
=============================================================
* Reads raw Parquet grids from cache/raw_parquet/
* Builds a per-patient binary ON/OFF sequence (hourly, all days)
* Fits a 2-state Multinomial HMM  (OFF ↔ ON hidden states)
* Uses the flattened 2×2 transition matrix + 2-element emission
  distribution as a 6-D feature vector
* Runs the usual k-scan (WCSS, Silhouette, CH, DB) → pick k
* Clusters with K-means; saves metric plots & cluster plots

Requires:  numpy  pandas  matplotlib  tqdm  scikit-learn  hmmlearn  scipy
"""

from __future__ import annotations
import sys, random, math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from hmmlearn.hmm import CategoricalHMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ───── paths / constants ────────────────────────────────────────
CACHE_RAW   = Path("cache/raw_parquet")
PLOT_PAT    = Path("plots/patients")        # for cached PNGs
OUT_ROOT    = Path("output_plots")
SEED        = 7
random.seed(SEED); np.random.seed(SEED)

# ───── helper: load 24×day matrix from Parquet ──────────────────
def load_day_bin(pid: str) -> np.ndarray:
    """Return binary matrix (n_days x 24) using same inclusive rule as stage-1."""
    pq = CACHE_RAW / f"{pid}.parquet"
    df = pd.read_parquet(pq)
    n_rows, n_days = df.shape
    day_bin = np.zeros((n_days, 24), dtype=np.uint8)
    # reuse t2h / _mark_full_hour from stage-1 without import
    def t2h(t): return t.hour + t.minute/60 + t.second/3600
    from datetime import time
    def to_time(val):
        if pd.isna(val): return None
        if isinstance(val, time): return val
        try: return pd.to_datetime(val).time()
        except: return None
    for r in range(0, n_rows, 2):
        if r+1 >= n_rows: break
        on_row, off_row = df.iloc[r], df.iloc[r+1]
        for d in range(n_days):
            on, off = to_time(on_row[d]), to_time(off_row[d])
            if on is None or off is None: continue
            s,e = t2h(on), t2h(off)
            if e < s:
                _mark(day_bin[d], s,24)
                if d+1 < n_days: _mark(day_bin[d+1], 0, e)
            else:
                _mark(day_bin[d], s, e)
    return day_bin

def _mark(row, s, e):
    for h in range(int(math.floor(s)), int(math.floor(e))+1):
        if 0<=h<24: row[h]=1

# ───── build HMM features ───────────────────────────────────────
def hmm_features(seq: np.ndarray, seed=SEED) -> np.ndarray:
    """
    Fit a 2-state Categorical HMM on a 0/1 sequence.
    Return 6-D vector [A11 A12 A21 A22  B0  B1].
    """
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)           # shape (T, 1) of ints 0/1

    model = CategoricalHMM(
        n_components=2,
        n_features=2,                    # symbols = {0,1}
        random_state=seed,
        n_iter=100,
        tol=1e-4,
        init_params="",                    # keep our manual init
        params="ste",                      # learn start, trans, emit
    )
    # Initialise with a slight bias
    model.startprob_    = np.array([0.5, 0.5])
    model.transmat_     = np.array([[0.9, 0.1],
                                    [0.1, 0.9]])
    model.emissionprob_ = np.array([[1.0, 0.0],   # state 0 → emit 0
                                    [0.0, 1.0]])  # state 1 → emit 1
    model.fit(seq)

    A = model.transmat_.ravel()           # 4 elements
    B = model.emissionprob_[:, 1]         # P(emit 1) for each state
    return np.concatenate([A, B]).astype(np.float32)

# ───── k-scan & plots (same as earlier) ─────────────────────────
def metric_scan(Z, kmin=2, kmax=15):
    ks,wcss,sil,ch,db=[],[],[],[],[]
    for k in range(kmin,kmax+1):
        km = KMeans(n_clusters=k, random_state=SEED).fit(Z)
        lab = km.labels_
        ks.append(k); wcss.append(km.inertia_)
        if len(np.unique(lab))>1:
            sil.append(silhouette_score(Z,lab))
            ch.append(calinski_harabasz_score(Z,lab))
            db.append(davies_bouldin_score(Z,lab))
        else:
            sil.append(np.nan); ch.append(np.nan); db.append(np.nan)
    return ks,np.array(wcss),np.array(sil),np.array(ch),np.array(db)

def choose_k(ks,wcss,sil,ch,db):
    r = sil.argsort()[::-1].argsort()+ch.argsort()[::-1].argsort()+db.argsort().argsort()
    knee = ks[np.argmin(np.diff(np.diff(wcss)))+2]
    best = ks[int(r.argmin())]
    return knee if best==2 and knee!=2 else best

def save_metric_plot(ks,y,title,out,color="b"):
    fig,ax=plt.subplots(); ax.plot(ks,y,"o-",color=color)
    ax.set(xlabel="k",ylabel=title.split()[0],title=title); ax.grid(True)
    fig.savefig(out/f"{title}.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ───── main ─────────────────────────────────────────────────────
def main():
    run = ensure(OUT_ROOT / f"hmm_{datetime.now():%Y%m%d_%H%M%S}")
    pqs  = sorted(CACHE_RAW.glob("*.parquet"))
    if not pqs:
        sys.exit("No raw Parquet files; run cache builder first.")

    feats = []; ids = []
    print("Fitting HMM per patient …")
    for pq in tqdm(pqs):
        pid = pq.stem
        day_bin = load_day_bin(pid)
        seq = day_bin.flatten()   # concat days -> T×1
        if seq.sum()==0:          # all OFF: skip
            continue
        feats.append(hmm_features(seq))
        ids.append(pid)
    X = np.vstack(feats)
    print("Feature matrix",X.shape)

    # k-scan
    ks,wcss,sil,ch,db = metric_scan(X,2,15)
    for name,y,col in [("elbow_WCSS",wcss,"b"),
                       ("silhouette",sil,"g"),
                       ("calinski_harabasz",ch,"m"),
                       ("davies_bouldin",db,"c")]:
        save_metric_plot(ks,y,name,run,col)
    best_k = choose_k(ks,wcss,sil,ch,db)
    print("Chosen k=",best_k)

    # final clustering
    lab = KMeans(n_clusters=best_k, random_state=SEED).fit_predict(X)

    # simple cluster size bar
    unique,counts = np.unique(lab,return_counts=True)
    fig,ax=plt.subplots(); ax.bar(unique,counts,color="skyblue")
    ax.set(title="Cluster sizes (HMM features)",xlabel="Cluster",ylabel="#patients")
    save_fig(fig,"cluster_sizes",run)

    # optional: save assignments
    np.save(run/"cluster_labels.npy",lab)
    with (run/"patient_ids.txt").open("w") as f:
        for pid in ids: f.write(pid+"\n")

    print("✅ Finished. Outputs in",run)

# ───── util save_fig & ensure ───────────────────────────────────
def save_fig(fig,name,out): fig.savefig(out/f"{name}.png",dpi=300,bbox_inches="tight"); plt.close(fig)
def ensure(p:Path)->Path: p.mkdir(parents=True,exist_ok=True); return p

if __name__ == "__main__":
    main()