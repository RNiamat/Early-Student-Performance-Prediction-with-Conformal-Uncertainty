"""Regenerate the summary figures from results/*.csv (no raw data needed).

Usage (from the repo root):
    python scripts/make_figures.py

Only quantities that exist in the results files are drawn. Error bars are shown only where a
spread was actually saved: the SD over folds (cross-validation) or the SD over the 20 splits.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

SCEN = ["A", "B", "C"]
LABELS = ["A\nbackground", "B\n+ G1", "C\n+ G1 + G2"]
BLUE, ORANGE, GREY, RED = "#3b6ea8", "#d9822b", "#6b6b6b", "#c0392b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

ms = pd.read_csv(RES / "multiseed_summary.csv").set_index("scenario").loc[SCEN]
t42 = pd.read_csv(RES / "test_results_seed42.csv")
cov42 = pd.read_csv(RES / "coverage_seed42.csv").set_index("scenario").loc[SCEN]
cv = pd.read_csv(RES / "cv_results.csv")
err42 = pd.read_csv(RES / "error_analysis_seed42.csv").set_index("scenario").loc[SCEN]
x = np.arange(3)

# ---- Figure 1: how much does earlier information help? (20 splits) -------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
ax.errorbar(x, ms["mae_mean"], yerr=ms["mae_std"], fmt="o-", color=BLUE, capsize=5, lw=2,
            label="Random Forest, mean ± SD over 20 splits")
ax.axhline(ms["baseline_mae"].iloc[0], color=GREY, ls="--", label="Mean baseline (20-split mean)")
rf42 = t42[t42.selected == "yes"].set_index("scenario").loc[SCEN, "mae"]
ax.plot(x, rf42, "D", mfc="none", color=ORANGE, ms=8, label="Seed-42 split (single)")
ax.set_xticks(x, LABELS); ax.set_ylim(0, 4); ax.set_ylabel("Test MAE (grade points)")
ax.set_title("Error vs information available"); ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
ax.vlines(x, ms["cov_min"] * 100, ms["cov_max"] * 100, color=BLUE, alpha=0.35, lw=8, label="min to max over 20 splits")
ax.errorbar(x, ms["cov_mean"] * 100, yerr=ms["cov_std"] * 100, fmt="o", color=BLUE, capsize=5, lw=2, label="mean ± SD")
ax.plot(x, cov42["coverage_all"] * 100, "D", mfc="none", color=ORANGE, ms=8, label="Seed-42 split (single)")
ax.axhline(90, color=GREY, ls="--", label="90% target")
ax.set_xticks(x, LABELS); ax.set_ylim(70, 100); ax.set_ylabel("Empirical coverage (%)")
ax.set_title("Conformal coverage (y-axis starts at 70%)"); ax.legend(fontsize=8, loc="lower right")

ax = axes[2]
ax.bar(x, ms["width_mean"], color=BLUE, width=0.55)
ax.axhline(20, color=GREY, ls="--"); ax.text(2.45, 20.3, "full grade scale (20)", ha="right", fontsize=9, color=GREY)
for xi, w in zip(x, ms["width_mean"]):
    ax.text(xi, w + 0.4, f"{w:.1f}", ha="center", fontsize=10)
ax.set_xticks(x, LABELS); ax.set_ylim(0, 23); ax.set_ylabel("Mean interval width (grade points)")
ax.set_title("Interval width (20-split mean)")

fig.suptitle("20 resplits of the same 395 students: spreads are not independent-sample errors", fontsize=10, y=0.02)
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(FIG / "fig_how_early_20splits.png", dpi=150, bbox_inches="tight"); plt.close()

# ---- Figure 2: cross-validation (train set only, 5 folds) ----------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.6))
models = ["Baseline (mean)", "Ridge", "RandomForest"]
cols = [GREY, ORANGE, BLUE]
for i, (m, c) in enumerate(zip(models, cols)):
    d = cv[cv.model == m].set_index("scenario").loc[SCEN]
    ax.errorbar(x + (i - 1) * 0.18, d["mae_mean"], yerr=d["mae_sd"], fmt="o", color=c, capsize=4, lw=1.8, label=m)
ax.set_xticks(x, LABELS); ax.set_ylim(0, 4.5); ax.set_ylabel("CV MAE (grade points)")
ax.set_title("5-fold CV on the training set (untuned models), mean ± SD over folds")
ax.legend(fontsize=9, loc="lower left")
plt.tight_layout(); plt.savefig(FIG / "fig_cv_results.png", dpi=150, bbox_inches="tight"); plt.close()

# ---- Figure 3: zero-grade students -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
w = 0.36
ax = axes[0]
ax.bar(x - w / 2, err42["mae_g3_pos"], w, color=BLUE, label=f"G3 > 0 (n = {int(err42['n_g3_pos'].iloc[0])})")
ax.bar(x + w / 2, err42["mae_g3_zero"], w, color=RED, label=f"G3 = 0 (n = {int(err42['n_g3_zero'].iloc[0])})")
for xi, a, b in zip(x, err42["mae_g3_pos"], err42["mae_g3_zero"]):
    ax.text(xi - w / 2, a + 0.12, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(xi + w / 2, b + 0.12, f"{b:.2f}", ha="center", fontsize=9)
ax.set_xticks(x, LABELS); ax.set_ylabel("Test MAE (grade points)"); ax.legend(fontsize=9)
ax.set_title("Seed-42 test error by group (single split)")

ax = axes[1]
ax.bar(x - w / 2, ms["cov_pos_mean"] * 100, w, color=BLUE, label="G3 > 0")
ax.bar(x + w / 2, ms["cov_zero_mean"] * 100, w, color=RED, label="G3 = 0 (7 or 8 students per split)")
for xi, a, b in zip(x, ms["cov_pos_mean"], ms["cov_zero_mean"]):
    ax.text(xi - w / 2, a * 100 + 1.5, f"{a * 100:.1f}", ha="center", fontsize=9)
    ax.text(xi + w / 2, b * 100 + 1.5, f"{b * 100:.1f}", ha="center", fontsize=9)
ax.axhline(90, color=GREY, ls="--", label="90% target")
ax.set_xticks(x, LABELS); ax.set_ylim(0, 138); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Mean coverage over 20 splits (%)")
ax.legend(fontsize=9, loc="upper center", ncol=1); ax.set_title("Conformal coverage by group (20-split mean)")
plt.tight_layout(); plt.savefig(FIG / "fig_zero_grade_effect.png", dpi=150, bbox_inches="tight"); plt.close()

print("wrote fig_how_early_20splits.png, fig_cv_results.png, fig_zero_grade_effect.png")
