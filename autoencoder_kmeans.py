"""
==============================================================
        Auto‑CPAP  ‑  Latent‑Clustering   (stage‑2 script)
==============================================================
Assumes stage‑1 (`build_cached_patient_data.py`) has filled:
    cache/binned/              ← .npy   (24‑D vectors)
    plots/patients/            ← actograms / dists  (png)

Outputs for *each run* go to:
    output_plots/sample_N/
        ├── metric_plots  (elbow, silhouette, …)
        ├── spaghetti_*.png
        ├── boxplot_*.png
        └── mean_spaghetti_all_clusters.png
"""

from __future__ import annotations
import os, sys, random, math, glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from tqdm import tqdm                     # ← NEW

# ───────────────────────── paths & seed ──────────────────────────
CACHE_BINNED_DIR = Path("cache/binned")
OUTPUT_FOLDER    = Path("output_plots")
SEED = 7
random.seed(SEED)
np.random.seed(SEED)

# ────────────────────────── helpers ──────────────────────────────
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def new_run_folder(base: Path) -> Path:
    ensure_dir(base)
    existing = [
        int(f.name.split("_")[1])
        for f in base.glob("sample_*")
        if f.name.split("_")[1].isdigit()
    ]
    run = base / f"sample_{max(existing) + 1 if existing else 1}"
    run.mkdir()
    return run

def save_fig(fig, fname, folder):
    fig.savefig(folder / f"{fname}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

# ─────────────────── load cached vectors ────────────────────────
def load_usage_vectors() -> tuple[np.ndarray, list[str]]:
    files = sorted(CACHE_BINNED_DIR.glob("*.npy"))
    if not files:
        sys.exit("❌  No .npy vectors found in cache/binned/. Run stage‑1 first.")

    vecs, ids = [], []
    for fp in tqdm(files, desc="Loading cached vectors"):
        vecs.append(np.load(fp))
        ids.append(fp.stem)

    X = np.vstack(vecs).astype(np.float32)
    print(f"Loaded {X.shape[0]} patients × {X.shape[1]} dims.")
    return X, ids

# ─────────────────── autoencoder training ───────────────────────
def train_autoencoder(X: np.ndarray, latent_dim=8, epochs=100, batch=16):
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    inp = Input(shape=(X.shape[1],))
    h1  = Dense(16, activation="relu")(inp)
    lat = Dense(latent_dim, activation="relu")(h1)
    h2  = Dense(16, activation="relu")(lat)
    out = Dense(X.shape[1], activation="sigmoid")(h2)

    ae = Model(inp, out)
    ae.compile(Adam(1e-3), loss="mse")
    ae.fit(X, X, epochs=epochs, batch_size=batch, shuffle=True, verbose=1)

    # encoder
    from tensorflow.keras import Input as KInput
    encoder = Model(inp, lat)
    latent_input = KInput(shape=(latent_dim,))
    dec_out = ae.layers[-1](ae.layers[-2](latent_input))
    decoder = Model(latent_input, dec_out)
    return encoder, decoder

# ───────────── metric sweep and suggestion ──────────────────────
def metric_scan(Z, kmin=2, kmax=15):
    ks, wcss, sil, ch, db = [], [], [], [], []
    for k in tqdm(range(kmin, kmax + 1), desc="Scanning k"):
        km = KMeans(n_clusters=k, random_state=SEED).fit(Z)
        labels = km.labels_
        ks.append(k)
        wcss.append(km.inertia_)

        if len(np.unique(labels)) > 1:
            sil.append(silhouette_score(Z, labels))
            ch.append(calinski_harabasz_score(Z, labels))
            db.append(davies_bouldin_score(Z, labels))
        else:
            sil.append(np.nan)
            ch.append(np.nan)
            db.append(np.nan)
    return ks, np.array(wcss), np.array(sil), np.array(ch), np.array(db)

def suggest_k(ks, wcss, sil, ch, db) -> int:
    # rank‑sum heuristic
    r_sil = sil.argsort()[::-1].argsort()
    r_ch  = ch.argsort()[::-1].argsort()
    r_db  = db.argsort().argsort()
    d2 = np.diff(np.diff(wcss))
    knee_k = ks[np.argmin(d2) + 2] if len(d2) else ks[-1]
    rank_sum = r_sil + r_ch + r_db
    best = ks[int(rank_sum.argmin())]
    return knee_k if best == 2 and knee_k != 2 else best

# ───────────── plotting metric curves ───────────────────────────
def plot_metrics(ks, wcss, sil, ch, db, folder):
    def _plot(y, ylab, name, color="b"):
        fig, ax = plt.subplots()
        ax.plot(ks, y, "o-", color=color)
        ax.set(xlabel="k", ylabel=ylab, title=name)
        ax.grid(True)
        save_fig(fig, name, folder)

    _plot(wcss, "WCSS", "elbow_WCSS")
    _plot(sil,  "Silhouette", "silhouette", "g")
    _plot(ch,   "Calinski‑Harabasz", "calinski_harabasz", "m")
    _plot(db,   "Davies‑Bouldin", "davies_bouldin", "c")

# ─────────── cluster‑level plotting helpers ─────────────────────
def spaghetti(labels, curves, out_dir):
    hrs = np.arange(24)
    for cl in tqdm(np.unique(labels), desc="Spaghetti plots"):
        idx  = np.where(labels == cl)[0]
        prof = curves[idx]
        mean = prof.mean(axis=0)

        fig, ax = plt.subplots(figsize=(8, 4))
        for c in prof:
            ax.plot(hrs, c, color="gray", alpha=0.3)
        ax.plot(hrs, mean, "o-b", lw=2, label="Mean")
        ax.set(xlim=[0, 23], xticks=hrs, xlabel="Hour",
               ylabel="Usage", title=f"Spaghetti (Cluster {cl})")
        ax.legend()
        ax.grid(True)
        save_fig(fig, f"spaghetti_cluster_{cl}", out_dir)

def mean_spaghetti(labels, curves, out_dir):
    hrs = np.arange(24)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cl in np.unique(labels):
        idx  = np.where(labels == cl)[0]
        mean = curves[idx].mean(axis=0)
        ax.plot(hrs, mean, "o-", lw=2, label=f"Cluster {cl}")
    ax.set(xlim=[0, 23], xticks=hrs, xlabel="Hour",
           ylabel="Mean usage", title="Mean curves by cluster")
    ax.grid(True)
    ax.legend()
    save_fig(fig, "mean_spaghetti_all_clusters", out_dir)

def boxplots(X24, labels, out_dir):
    hrs = np.arange(24)
    for cl in tqdm(np.unique(labels), desc="Boxplots"):
        idx = np.where(labels == cl)[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot([X24[idx, h] for h in hrs], positions=hrs, widths=0.6)
        ax.set(xlim=[-0.5, 23.5], xticks=hrs, xlabel="Hour",
               ylabel="Usage", title=f"Box‑Whisker (Cluster {cl})")
        ax.grid(axis="y")
        save_fig(fig, f"boxplot_cluster_{cl}", out_dir)

# ─────────────────────────── main flow ───────────────────────────
def main():
    run = new_run_folder(OUTPUT_FOLDER)
    print("→ output:", run)

    X24, patient_ids = load_usage_vectors()

    mask = ~np.isnan(X24).all(axis=1)      # keep rows with at least one real number
    X24     = X24[mask]
    patient_ids = [pid for pid, keep in zip(patient_ids, mask) if keep]

    print(f"Training on {X24.shape[0]} patients (dropped {mask.size - mask.sum()}).")

    encoder, decoder = train_autoencoder(X24, latent_dim=8, epochs=100, batch=16)
    Z = encoder.predict(X24)
    print("Latent shape:", Z.shape)

    ks, wcss, sil, ch, db = metric_scan(Z, kmin=2, kmax=15)
    plot_metrics(ks, wcss, sil, ch, db, run)

    suggested = suggest_k(ks, wcss, sil, ch, db)
    print(f"\nSuggested k == {suggested}")
    user = input("Press Enter to accept, or type another k (or 'end' to quit): ").strip().lower()
    if user == "end":
        sys.exit(0)
    if user:
        try:
            suggested = int(user)
        except ValueError:
            print("⚠️  Not an int; keeping suggestion.")

    print(f"\nRunning final KMeans with k = {suggested} …")
    km   = KMeans(n_clusters=suggested, random_state=SEED).fit(Z)
    lab  = km.labels_
    rec  = decoder.predict(Z)

    spaghetti(lab, rec, run)
    mean_spaghetti(lab, rec, run)
    boxplots(X24, lab, run)

    print("\n✅  All cluster plots written to", run)

if __name__ == "__main__":
    main()