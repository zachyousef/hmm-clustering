#!/usr/bin/env python3
from __future__ import annotations
# ╔═══════════════════════════════════════════════════════════════════╗
#  CHRONOTYPE + ADHERENCE CLUSTERING – 12-STATE HMM (multi-head)     #
# ╚═══════════════════════════════════════════════════════════════════╝
# QC knobs – tweak if desired
PATIENT_MODE, SAMPLE_N           = "sample", 100        # "all"| "sample"
MIN_DAYS, MIN_ACTIVE_DAYS        = 60, 50
MIN_MEAN_HRS, MIN_ADHERENCE_FRAC = 4.0, 0.0
# --------------------------------------------------------------------

import os, sys, random, warnings, math
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import time, datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from hmmlearn.hmm import CategoricalHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                              SpectralClustering, OPTICS)
from sklearn.mixture import BayesianGaussianMixture
import hdbscan, umap
from joblib import Parallel, delayed
from kneed import KneeLocator
from tslearn.metrics import cdist_dtw
from tslearn.clustering import TimeSeriesKMeans

SEED = 7
random.seed(SEED); np.random.seed(SEED)

# ─── constants ──────────────────────────────────────────────────────
PHASES, ADH_BANDS  = 4, 3                # 4×3 = 12 HMM states
N_STATES, UMAP_K   = PHASES*ADH_BANDS, 30
CACHE_RAW, OUT_BASE= Path("cache/raw_parquet"), Path("output_plots")

# ─── I/O helpers ────────────────────────────────────────────────────
def to_time(v):
    if pd.isna(v): return None
    if isinstance(v, time): return v
    try: return pd.to_datetime(v).time()
    except: return None
def t2h(t): return t.hour + t.minute/60 + t.second/3600
def _mark(row, s, e):                          # binary mask helper
    for h in range(int(s), int(e)+1):
        if 0<=h<24: row[h]=1
def load_day_bin(pid:str)->np.ndarray:
    df   = pd.read_parquet(CACHE_RAW/f"{pid}.parquet")
    rows,days = df.shape
    if rows==0 or days==0: return np.empty((0,24),np.uint8)
    db = np.zeros((days,24),np.uint8)
    for r in range(0, rows, 2):
        if r+1>=rows: break
        on_row,off_row=df.iloc[r],df.iloc[r+1]
        for d in range(days):
            on,off=to_time(on_row.iat[d]),to_time(off_row.iat[d])
            if on is None or off is None: continue
            s,e=t2h(on),t2h(off)
            if e<s:
                _mark(db[d],s,24)
                if d+1<days:_mark(db[d+1],0,e)
            else:_mark(db[d],s,e)
    return db

# ─── feature engineering ────────────────────────────────────────────
def hmm12_features(seq:np.ndarray)->np.ndarray:
    if seq.ndim==1: seq=seq.reshape(-1,1)
    block=np.full((PHASES,PHASES),0.05); np.fill_diagonal(block,0.85)
    trans=np.kron(np.eye(ADH_BANDS),block); trans/=trans.sum(1,keepdims=True)
    model=CategoricalHMM(N_STATES,2,random_state=SEED,
                         init_params="e",params="ste",
                         n_iter=300,tol=1e-4)
    model.startprob_=np.ones(N_STATES)/N_STATES
    model.transmat_ =trans
    model.fit(seq)
    z = model.transmat_.sum(1)==0
    if np.any(z): model.transmat_[z]=np.ones(N_STATES)/N_STATES
    return np.hstack([model.transmat_.ravel(),
                      model.emissionprob_[:,1]]).astype(np.float32)

def circadian_features(db:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    prof=db.mean(0)
    on  =np.argmax(db,1) if db.size else np.array([0])
    mid =np.angle(np.mean(np.exp(1j*2*np.pi*on/24)))*24/(2*np.pi)%24
    dur =db.sum(1).mean() if db.size else 0
    var =db.sum(0).var()  if db.size else 0
    return np.hstack([prof,[mid,dur,var]]).astype(np.float32), prof

def build_features(pid:str):
    db=load_day_bin(pid); n_days=db.shape[0]
    if n_days==0: return None
    active=np.count_nonzero(db.sum(1))
    mean_hrs=db.sum()/n_days
    adher=active/n_days
    if (n_days<MIN_DAYS or active<MIN_ACTIVE_DAYS
        or mean_hrs<MIN_MEAN_HRS or adher<MIN_ADHERENCE_FRAC):
        return None
    hmm_vec = hmm12_features(db.flatten())
    circ_vec, prof24 = circadian_features(db)
    return pid, np.hstack([hmm_vec, circ_vec]), prof24

# ─── clustering helpers ---------------------------------------------
def save_sizes(lbl,title,fname,out):
    lbl=lbl[lbl>=0]
    plt.figure(figsize=(4,3))
    plt.bar(range(len(np.bincount(lbl))),np.bincount(lbl))
    plt.xlabel("Cluster"); plt.ylabel("#"); plt.title(title)
    plt.tight_layout(); plt.savefig(out/fname,dpi=300); plt.close()

def kmeans_auto(X,out):
    ks=range(2,11); sil=[]
    for k in ks:
        km=KMeans(k,random_state=SEED,n_init="auto").fit(X)
        sil.append(silhouette_score(X,km.labels_))
    k_opt=ks[int(np.argmax(sil))]
    km=KMeans(k_opt,random_state=SEED,n_init="auto").fit(X)
    np.save(out/"labels_kmeans.npy",km.labels_)
    save_sizes(km.labels_,"K-means sizes","cluster_sizes_kmeans.png",out)
    return km.labels_

def hdbscan_auto(X,out):
    hdb=hdbscan.HDBSCAN(min_cluster_size=25,min_samples=10).fit(X)
    np.save(out/"labels_hdbscan.npy",hdb.labels_)
    save_sizes(hdb.labels_,"HDBSCAN sizes","cluster_sizes_hdbscan.png",out)
    return hdb.labels_

def dpgmm_auto(X,out):
    dpg=BayesianGaussianMixture(
        n_components=15,weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.05,random_state=SEED).fit(X)
    lbl=dpg.predict(X)
    np.save(out/"labels_dpgmm.npy",lbl)
    save_sizes(lbl,"DP-GMM sizes","cluster_sizes_dpgmm.png",out)
    return lbl

def tskmeans_auto(prof24,Xz,out):
    best_lbl,best_s=-1,-1
    for k in range(2,11):
        tsk=TimeSeriesKMeans(k,metric="dtw",n_init=2,random_state=SEED).fit(prof24)
        s=silhouette_score(Xz,tsk.labels_)
        if s>best_s: best_lbl,best_s=tsk.labels_,s
    np.save(out/"labels_tskmeans.npy",best_lbl)
    save_sizes(best_lbl,"TS-KMeans sizes","cluster_sizes_tskmeans.png",out)
    return best_lbl

def agglo_dtw_auto(prof24,out):
    D=cdist_dtw(prof24.astype(np.float32))
    best_lbl,best_s=-1,-1
    for k in range(2,11):
        ac=AgglomerativeClustering(k,metric="precomputed",linkage="average").fit(D)
        s=silhouette_score(D,ac.labels_,metric="precomputed")
        if s>best_s: best_lbl,best_s=ac.labels_,s
    np.save(out/"labels_agglo_dtw.npy",best_lbl)
    save_sizes(best_lbl,"Agglo-DTW sizes","cluster_sizes_agglo_dtw.png",out)
    return best_lbl

def spectral_rbf_auto(X,out):
    best_lbl,best_s=-1,-1
    for k in range(2,11):
        sc=SpectralClustering(k,affinity="rbf",random_state=SEED).fit(X)
        s=silhouette_score(X,sc.labels_)
        if s>best_s: best_lbl,best_s=sc.labels_,s
    np.save(out/"labels_spectral_rbf.npy",best_lbl)
    save_sizes(best_lbl,"Spectral-RBF sizes","cluster_sizes_spectral_rbf.png",out)
    return best_lbl

def optics_auto(X,out):
    op=OPTICS(min_samples=15,xi=0.05).fit(X)
    np.save(out/"labels_optics.npy",op.labels_)
    save_sizes(op.labels_,"OPTICS sizes","cluster_sizes_optics.png",out)
    return op.labels_

# ─── main ───────────────────────────────────────────────────────────
def main():
    if len(sys.argv)>2: sys.exit("usage: python chrono_cluster.py [output_dir]")
    out = Path(sys.argv[1]) if len(sys.argv)==2 else \
          OUT_BASE/f"chrono_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True,exist_ok=True)

    all_ids=sorted(p.stem for p in CACHE_RAW.glob("*.parquet"))
    ids=random.sample(all_ids,min(SAMPLE_N,len(all_ids))) \
        if PATIENT_MODE=="sample" else all_ids
    (out/"patient_ids_all.txt").write_text("\n".join(ids))

    print(f"Building features ({len(ids)} pts)…")
    res=Parallel(max(1,(os.cpu_count() or 2)-1))(
         delayed(build_features)(pid) for pid in tqdm(ids))

    kept,feats,prof24=[],[],[]
    for r in res:
        if r is None: continue
        pid,vec,prof=r
        kept.append(pid); feats.append(vec); prof24.append(prof)
    if not feats: sys.exit("❌  No patients passed QC.")
    print(f"QC passed: {len(kept)} / {len(ids)}")

    X   = np.vstack(feats).astype(np.float32)
    Xz  = StandardScaler().fit_transform(X)
    prof24 = np.vstack(prof24).astype(np.float32)
    (out/"patient_ids.txt").write_text("\n".join(kept))
    np.save(out/"features.npy",Xz)

    # ─── run all heads ────────────────────────────────────────────
    print("Clustering heads …")
    lbls={
        "kmeans"      : kmeans_auto(Xz,out),
        "hdbscan"     : hdbscan_auto(Xz,out),
        "dpgmm"       : dpgmm_auto(Xz,out),
        "tskmeans"    : tskmeans_auto(prof24,Xz,out),
        "agglo_dtw"   : agglo_dtw_auto(prof24,out),
        "spectral_rbf": spectral_rbf_auto(Xz,out),
        "optics"      : optics_auto(Xz,out)
    }

    print("\nSilhouette leaderboard")
    for n,lbl in lbls.items():
        try: s=silhouette_score(Xz,lbl) if len(np.unique(lbl))>1 else -1
        except: s=-1
        print(f"  {n:12s}: {s:5.3f}")

    emb=umap.UMAP(n_neighbors=UMAP_K,min_dist=0.1,random_state=SEED).fit_transform(Xz)
    plt.figure(figsize=(5,4.5))
    plt.scatter(emb[:,0],emb[:,1],c=lbls["kmeans"],cmap="tab10",
                s=6,alpha=0.8,lw=0)
    plt.title("UMAP – K-means colour")
    plt.tight_layout(); plt.savefig(out/"umap_kmeans.png",dpi=400); plt.close()
    print("✅  Artefacts in", out)

if __name__=="__main__":
    main()
