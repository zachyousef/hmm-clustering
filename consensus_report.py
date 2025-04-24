#!/usr/bin/env python3
"""
make_consensus_report.py
========================
Create a single PDF summarising consensus quality and the 10 patients
with the highest consensus scores.

Usage:
    python make_consensus_report.py  output_plots/consensus_20250421_233012
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# ─── helpers ────────────────────────────────────────────────────
def load_scores(run_dir: Path):
    """Return (scores , patient_ids) saved by consensus_clustering."""
    scores_npy = run_dir / "scores.npy"          # we saved this file below
    ids_txt    = run_dir / "patient_ids.txt"
    if not scores_npy.exists() or not ids_txt.exists():
        sys.exit("Consensus run missing expected scores.npy / patient_ids.txt")
    scores = np.load(scores_npy)
    with ids_txt.open() as f:
        ids = [ln.strip() for ln in f]
    return scores, ids

def img(path: Path):
    return Image.open(path)

def add_image(ax, im: Image.Image):
    ax.imshow(im); ax.axis("off")

# ─── main ───────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        print("Usage: python make_consensus_report.py  <consensus_run_folder>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        sys.exit(f"{run_dir} not found")

    # ----- 1. load scores & pick top‑10 ------------------------------------
    scores, ids = load_scores(run_dir)
    top_idx = np.argsort(scores)[::-1][:10]      # highest first
    top_ids = [ids[i] for i in top_idx]

    # map cached patient PNG names
    pat_dir = Path("plots/patients")             # stage‑1 output
    acto_pat = {p.stem.replace("_actogram",""): p for p in pat_dir.glob("*_actogram.png")}
    dist_pat = {p.stem.replace("_distribution",""): p for p in pat_dir.glob("*_distribution.png")}

    # sanity check
    missing = [pid for pid in top_ids if pid not in acto_pat or pid not in dist_pat]
    if missing:
        print("Warning: cached PNGs missing for", missing)

    # ----- 2. open PDF ------------------------------------------------------
    pdf_path = run_dir / "Consensus_Report.pdf"
    with PdfPages(pdf_path) as pdf:

        # Page 1 – heat‑map + hists stacked
        fig, axes = plt.subplots(3,1, figsize=(8.3, 11.7))   # A4 portrait
        for ax, png in zip(axes, ["Consensus_Matrix.png",
                                  "Consensus_OffDiag_Hist.png",
                                  "Consensus_Scores_Hist.png"]):
            add_image(ax, img(run_dir / png))
        pdf.savefig(fig); plt.close(fig)

        # Pages 2‑11 – one patient each
        for pid in top_ids:
            fig, axes = plt.subplots(1,2, figsize=(8.3, 11.7/2))
            add_image(axes[0], img(acto_pat[pid]))
            add_image(axes[1], img(dist_pat[pid]))
            fig.suptitle(f"Patient: {pid}   |   Consensus score = {scores[ids.index(pid)]:.3f}",
                         fontsize=14)
            pdf.savefig(fig); plt.close(fig)

    print("✅ PDF created:", pdf_path)

# -----------------------------------------------------------------
if __name__ == "__main__":
    main()