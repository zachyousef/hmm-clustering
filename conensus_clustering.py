#!/usr/bin/env python3
"""
=============================================================
        Auto-CPAP  –  Consensus Clustering  (stage-3)
=============================================================
Requirements (same env as previous stages):
    numpy  pandas  matplotlib  tqdm  scikit-learn  scipy  tensorflow==2.*
Assumes stage-1 has populated
    cache/binned/    ← 24-D .npy vectors
"""

from __future__ import annotations
import os, sys, random, glob
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list

# ───────── config ──────────────────────────────────────────────
CACHE_DIR   = Path("cache/binned")
OUT_ROOT    = Path("output_plots")
BASE_SEED   = 7
N_RUNS      = 10
LATENT_DIM  = 8
AE_EPOCHS   = 100
BATCH_SIZE  = 16

# ───────── utils ───────────────────────────────────────────────
def ensure(p: Path)->Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_fig(fig, name, folder):
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

def load_vectors():
    files = sorted(CACHE_DIR.glob("*.npy"))
    if not files:
        sys.exit("❌ No cached vectors found; run stage-1 first.")
    vecs, ids = [], []
    for fp in files:
        v = np.load(fp)
        vecs.append(v); ids.append(fp.stem)
    X = np.vstack(vecs).astype(np.float32)
    # drop all-NaN rows
    mask = ~np.isnan(X).all(axis=1)
    if not mask.all():
        print(f"Dropping {mask.size - mask.sum()} all-NaN vectors")
        X = X[mask]; ids = [i for i,m in zip(ids,mask) if m]
    return X, ids

def train_autoencoder(X, latent_dim=8, epochs=100, batch=16, seed=0):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    # reproducible
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    inp = Input(shape=(X.shape[1],))
    x = Dense(16, activation="relu")(inp)
    lat = Dense(latent_dim, activation="relu")(x)
    x = Dense(16, activation="relu")(lat)
    out = Dense(X.shape[1], activation="sigmoid")(x)
    ae = Model(inp, out)
    ae.compile(Adam(1e-3), loss="mse")
    ae.fit(X, X, epochs=epochs, batch_size=batch, shuffle=True, verbose=0)
    encoder = Model(inp, lat)
    return encoder

def scan_k(Z, kmin=2, kmax=15, seed=0):
    ks, wcss, sil, ch, db = [], [], [], [], []
    for k in range(kmin, kmax+1):
        km = KMeans(n_clusters=k, random_state=seed).fit(Z)
        lbl = km.labels_
        ks.append(k); wcss.append(km.inertia_)
        if len(np.unique(lbl)) > 1:
            sil.append(silhouette_score(Z, lbl))
            ch.append(calinski_harabasz_score(Z, lbl))
            db.append(davies_bouldin_score(Z, lbl))
        else:
            sil.append(np.nan); ch.append(np.nan); db.append(np.nan)
    return ks, np.array(wcss), np.array(sil), np.array(ch), np.array(db)

def choose_k(ks, wcss, sil, ch, db):
    r = sil.argsort()[::-1].argsort() + ch.argsort()[::-1].argsort() + db.argsort().argsort()
    knee = ks[np.argmin(np.diff(np.diff(wcss)))+2]
    best = ks[int(r.argmin())]
    return knee if best==2 and knee!=2 else best

def plot_metric(ks, y, title, ylabel, out, color="b"):
    fig, ax = plt.subplots(); ax.plot(ks, y, "o-", color=color)
    ax.set(xlabel="k", ylabel=ylabel, title=title); ax.grid(True)
    save_fig(fig, title, out)

# ───────── main ────────────────────────────────────────────────
def main():
    out_dir = ensure(OUT_ROOT / f"consensus_{datetime.now():%Y%m%d_%H%M%S}")
    print("→ output:", out_dir)

    X24, ids = load_vectors()
    print("Using", X24.shape[0], "patients.")

    # ── pilot run to pick k ────────────────────────────────────
    enc0 = train_autoencoder(X24, LATENT_DIM, AE_EPOCHS, BATCH_SIZE, BASE_SEED)
    Z0   = enc0.predict(X24)
    ks, wcss, sil, ch, db = scan_k(Z0, 2, 15, BASE_SEED)
    plot_metric(ks, wcss, "elbow_WCSS", "WCSS", out_dir)
    plot_metric(ks, sil , "silhouette", "Silhouette", out_dir, "g")
    plot_metric(ks, ch  , "calinski_harabasz", "CH", out_dir, "m")
    plot_metric(ks, db  , "davies_bouldin", "DB", out_dir, "c")
    best_k = choose_k(ks, wcss, sil, ch, db)
    print(f"Chosen best_k = {best_k}")

    # ── consensus runs ─────────────────────────────────────────
    n = X24.shape[0]
    C = np.zeros((n, n), dtype=float)

    for r in tqdm(range(N_RUNS), desc="Consensus runs"):
        seed = BASE_SEED + 100*r
        enc  = train_autoencoder(X24, LATENT_DIM, AE_EPOCHS, BATCH_SIZE, seed)
        Z    = enc.predict(X24)
        lab  = KMeans(n_clusters=best_k, random_state=seed).fit_predict(Z)
        for i in range(n):
            C[i, lab==lab[i]] += 1         # add 1 for all mates incl self
    C /= N_RUNS

    # ── reorder & plot heat-map ───────────────────────────────
    dist   = 1 - C
    order  = leaves_list(linkage(squareform(dist), method="average"))
    Cres   = C[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(Cres, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Consensus Matrix (reordered)"); ax.axis("off")
    fig.colorbar(im, ax=ax, label="Co-cluster fraction")
    save_fig(fig, "Consensus_Matrix", out_dir)

    # ── histogram of off-diagonal values ──────────────────────
    off = C[~np.eye(n, dtype=bool)]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(off, bins=20, color="skyblue", edgecolor="black")
    ax.set(title="Off-diagonal consensus distribution",
           xlabel="Co-cluster fraction", ylabel="Count")
    save_fig(fig, "Consensus_OffDiag_Hist", out_dir)

    # ── per-patient consensus score & its histogram ───────────
    scores = (C.sum(axis=1) - 1) / (n-1)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(scores, bins=20, color="lightgreen", edgecolor="black")
    ax.set(title="Patient consensus scores",
           xlabel="Mean co-cluster fraction", ylabel="Frequency")
    save_fig(fig, "Consensus_Scores_Hist", out_dir)

    np.save(out_dir / "scores.npy", scores)
    with (out_dir / "patient_ids.txt").open("w") as f:
        for pid in ids:
            f.write(pid + "\n")
        
    print("✅  Finished - artefacts in", out_dir)

if __name__ == "__main__":
    main()