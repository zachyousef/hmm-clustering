#!/usr/bin/env python3
from __future__ import annotations
"""
classic-layout PDF reporter  – 2025-04-27
----------------------------------------
usage
  python chrono_report.py  <run_dir> [algo|all|list]

if [algo] omitted → first available in preference order
"""
import sys, io, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader

THREADS    = 24
PAT_DIR    = Path("plots/patients")
PREF_ORDER = ["kmeans","dpgmm","hdbscan",
              "tskmeans","agglo_dtw","spectral_rbf","optics"]

PAGE_W, PAGE_H = letter
LM, RM, TM, BM = 36, 36, 36, 36          # margins

# ───── helpers ──────────────────────────────────────────────────────
def load_png(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")

def centroid_nearest(X: np.ndarray, labels: np.ndarray, n=3):
    uniq, near = [], {}
    for cl in sorted(set(labels[labels>=0])):
        idx = np.where(labels==cl)[0]
        c   = X[idx].mean(0)
        d   = np.linalg.norm(X[idx]-c, axis=1)
        near[cl] = idx[np.argsort(d)][:n]
        uniq.append(cl)
    return uniq, near

def draw_center(pdf: Canvas, img: Image.Image,
                x: float, y: float, w_fit: float, h_fit: float):
    scale = min(w_fit/img.width, h_fit/img.height)
    w, h  = img.width*scale, img.height*scale
    buf = io.BytesIO(); img.save(buf,"PNG"); buf.seek(0)
    pdf.drawImage(ImageReader(buf),
                  x + (w_fit-w)/2,
                  y + (h_fit-h)/2,
                  w, h)

# ───── single PDF builder ──────────────────────────────────────────
def build_report(run: Path, algo: str):
    lab = run/f"labels_{algo}.npy"
    if not lab.exists():
        print(f"⚠️  {lab.name} missing – skipped")
        return

    labels = np.load(lab)
    ids    = [ln.strip() for ln in open(run/"patient_ids.txt")]
    Xz     = np.load(run/"features.npy")
    uniq, near = centroid_nearest(Xz, labels)

    # collect all needed PNGs
    want=set(run.glob("*.png"))
    if (run/"umap_kmeans.png").exists(): want.add(run/"umap_kmeans.png")
    for cl in uniq:
        for i in near[cl]:
            pid=ids[i]
            want |= { PAT_DIR/f"{pid}_actogram.png",
                      PAT_DIR/f"{pid}_distribution.png" }
    want = [p for p in want if p.exists()]

    # threaded load
    imgs={}
    with ThreadPoolExecutor(THREADS) as pool:
        fut = {pool.submit(load_png,p):p for p in want}
        for f in as_completed(fut): imgs[fut[f]]=f.result()

    pdf_path = run/f"{algo}_Chronotype_Report.pdf"
    pdf = Canvas(str(pdf_path), pagesize=letter)

    # ---------- PAGE 1 : metrics / UMAP (2 cols) -------------------
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(LM, PAGE_H-TM+6, f"Chronotype report  –  {algo.upper()}")

    col_w = (PAGE_W - LM - RM) / 2
    row_h = 180
    x0, y0 = LM, PAGE_H - TM - 30    # start below title
    n = 0
    for p in sorted([p for p in want if p.parent==run]):
        cx = x0 + (n%2)*col_w
        cy = y0 - (n//2)*row_h - row_h
        draw_center(pdf, imgs[p], cx, cy, col_w, row_h-10)
        n += 1
        if cy < BM: break
    pdf.showPage()

    # ---------- cluster pages -------------------------------------
    row_h = 220
    img_w = (PAGE_W - LM - RM - 24) / 2   # gap 24
    img_h = row_h - 30                    # leave caption space

    for cl in uniq:
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(LM, PAGE_H-TM+6, f"Cluster {cl} ({algo}) – 3 nearest")

        y = PAGE_H - TM - 30
        for idx in near[cl]:
            pid = ids[idx]
            act  = imgs.get(PAT_DIR/f"{pid}_actogram.png")
            dist = imgs.get(PAT_DIR/f"{pid}_distribution.png")
            if act:
                draw_center(pdf, act , LM           , y-row_h+10, img_w, img_h)
            if dist:
                draw_center(pdf, dist, LM+img_w+24 , y-row_h+10, img_w, img_h)
            pdf.setFont("Helvetica", 10)
            pdf.drawString(LM, y-15, pid)
            y -= row_h
        pdf.showPage()

    pdf.save()
    print("✅", pdf_path.name, "created")

# ───── CLI entry ----------------------------------------------------
def main():
    if len(sys.argv) not in (2,3):
        sys.exit("usage: python chrono_report.py <run_dir> [algo|all|list]")
    run=Path(sys.argv[1])

    if len(sys.argv)==3:
        opt=sys.argv[2].lower()
        algos = PREF_ORDER if opt=="all" else [a.strip() for a in opt.split(",")]
    else:
        algos=[a for a in PREF_ORDER if (run/f"labels_{a}.npy").exists()][:1]
        if not algos: sys.exit("no labels_*.npy in run folder")

    for algo in algos: build_report(run, algo)

if __name__=="__main__":
    main()
