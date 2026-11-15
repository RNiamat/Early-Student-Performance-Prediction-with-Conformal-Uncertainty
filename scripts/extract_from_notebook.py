"""Rebuild results/*.csv and notebook figures from the SAVED OUTPUTS of the notebook.

The original result files (CSV checkpoints) were lost with the Kaggle session, so every
table here is parsed from text that the notebook printed. Values keep the precision that
was printed (2-3 decimals). Nothing is recomputed or estimated.

Usage (from the repo root):
    python scripts/extract_from_notebook.py
"""
import ast
import base64
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "early_student_performance_conformal.ipynb"
RES = ROOT / "results"
FIG = ROOT / "figures"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
CELLS = {}
for c in nb["cells"]:
    if c["cell_type"] == "code":
        m = re.search(r"\(was CELL (\S+)\)", "".join(c["source"]).split("\n")[0])
        if m:
            CELLS[m.group(1)] = c


def out_text(old_id):
    """All stream text printed by the cell that was called CELL <old_id> in the original."""
    cell = CELLS[old_id]
    return "".join("".join(o["text"]) for o in cell["outputs"] if o["output_type"] == "stream")


def rows(pattern, text, flags=re.M):
    found = re.findall(pattern, text, flags)
    if not found:
        raise ValueError(f"pattern not found: {pattern}")
    return found


SC = r"([ABC])"
NUM = r"(-?\d+\.\d+)"
MODEL = r"(Baseline \(mean\)|Ridge|RandomForest)"
tables = {}

# ---- data summary (cells 1.2, 1.3, 2.1, 2.2, 2.6) -------------------------------------
t12, t13, t21, t22, t26 = out_text("1.2"), out_text("1.3"), out_text("2.1"), out_text("2.2"), out_text("2.6")
nr, nc = re.search(r"Rows x Columns: \((\d+), (\d+)\)", t12).groups()
n_text = re.search(r"^object\s+(\d+)", t12, re.M).group(1)
n_int = re.search(r"^int64\s+(\d+)", t12, re.M).group(1)
n_miss = re.search(r"Total missing values: (\d+)", t13).group(1)
n_dup = re.search(r"Duplicate rows: (\d+)", t13).group(1)
g3 = {k: float(v) for k, v in re.findall(r"^(mean|std|min|max)\s+(-?\d+\.\d+)", t21, re.M)}
nz, ntot = re.search(r"Students with G3 = 0: (\d+) of (\d+)", t22).groups()
a0z, a0z_n = re.search(r"absences == 0 among G3 = 0 students: (\d+) of (\d+)", t26).groups()
a0p, a0p_n = re.search(r"absences == 0 among G3 > 0 students: (\d+) of (\d+)", t26).groups()
c_all = re.search(r"all students\s*: (-?\d+\.\d+)", t26).group(1)
c_pos = re.search(r"G3 > 0 only\s*: (-?\d+\.\d+)", t26).group(1)
data_summary = {
    "n_rows": nr, "n_columns": nc, "n_text_columns": n_text, "n_numeric_columns": n_int,
    "missing_values": n_miss, "duplicate_rows": n_dup,
    "g3_mean": g3["mean"], "g3_std": g3["std"], "g3_min": g3["min"], "g3_max": g3["max"],
    "n_zero_grade": nz, "n_zero_grade_total": ntot,
    "absences_zero_among_zero_grade": a0z, "n_zero_grade_for_absences": a0z_n,
    "absences_zero_among_positive": a0p, "n_positive_for_absences": a0p_n,
    "corr_absences_g3_all": c_all, "corr_absences_g3_positive_only": c_pos,
}
tables["data_summary"] = pd.DataFrame({"metric": list(data_summary), "value": list(data_summary.values())})

# ---- correlations with G3 (2.3) -----------------------------------------------------
t23 = out_text("2.3")
corr = rows(r"^(\w+)\s+(-?\d+\.\d+)$", t23.split("Correlation of each numeric feature with G3:")[1])
tables["g3_correlations"] = pd.DataFrame(corr, columns=["feature", "corr_with_g3"]).astype({"corr_with_g3": float})

# ---- split summary (4.2) ------------------------------------------------------------
t42 = out_text("4.2")
sp = rows(r"^(train|calib|test)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d+)\s+(\d+\.\d+)", t42)
tables["split_summary"] = pd.DataFrame(
    [(a, int(float(b)), float(c), int(float(d)), float(e), float(f), float(g)) for a, b, c, d, e, f, g in sp],
    columns=["set", "rows", "share_pct", "g3_zero_count", "g3_zero_share_pct", "g3_mean", "g3_std"])

# ---- CV results (6.4): "mean ± std" strings -----------------------------------------
t64 = out_text("6.4")
cv = rows(rf"^\s*{SC}\s+{MODEL}\s+{NUM} ± {NUM}\s+{NUM} ± {NUM}\s+{NUM} ± {NUM}", t64)
tables["cv_results"] = pd.DataFrame(
    cv, columns=["scenario", "model", "mae_mean", "mae_sd", "rmse_mean", "rmse_sd", "r2_mean", "r2_sd"]
).astype({k: float for k in ["mae_mean", "mae_sd", "rmse_mean", "rmse_sd", "r2_mean", "r2_sd"]})

# ---- tuning (7.3) -------------------------------------------------------------------
t73 = out_text("7.3")
tun = rows(rf"^\s*{SC}\s+(Ridge|RandomForest)\s+{NUM}\s+{NUM}\s+{NUM}\s+(\{{.*\}})", t73)
tables["tuning_results"] = pd.DataFrame(
    [(a, b, float(c), float(d), float(e), json.dumps(ast.literal_eval(p))) for a, b, c, d, e, p in tun],
    columns=["scenario", "model", "default_cv_mae", "tuned_cv_mae", "gain", "best_params"])

# ---- seed-42 test results (8.3) -----------------------------------------------------
t83 = out_text("8.3")
te = rows(rf"^\s*{SC}\s+{MODEL}\s+(yes\s+)?{NUM}\s+{NUM}\s+{NUM}\s*$", t83)
tables["test_results_seed42"] = pd.DataFrame(
    [(a, b, "yes" if c.strip() else "", float(d), float(e), float(f)) for a, b, c, d, e, f in te],
    columns=["scenario", "model", "selected", "mae", "rmse", "r2"])

# ---- bootstrap CIs (8.4) ------------------------------------------------------------
t84 = out_text("8.4")
bt = rows(rf"^\s*{SC}\s+RandomForest\s+{NUM}\s+\[{NUM}, {NUM}\]\s+{NUM}\s+{NUM}\s+\[{NUM}, {NUM}\]\s+(True|False)", t84)
tables["bootstrap_ci_seed42"] = pd.DataFrame(
    [(a, *map(float, r[:7]), r[7] == "True") for a, *r in bt],
    columns=["scenario", "test_mae", "mae_ci_low", "mae_ci_high", "baseline_mae", "improvement",
             "improvement_ci_low", "improvement_ci_high", "improvement_ci_excludes_0"])

# ---- q and width (9.3), coverage (10.1), q diagnostic (9.4) --------------------------
t93 = out_text("9.3")
qq = rows(rf"^\s*{SC}\s+{NUM}\s+(\d+)\s+{NUM}\s+{NUM}", t93)
tables["conformal_q_seed42"] = pd.DataFrame(
    [(a, float(b), int(c), float(d), float(e)) for a, b, c, d, e in qq],
    columns=["scenario", "q", "n_calib", "quantile_level", "avg_width_clipped"])

t101 = out_text("10.1")
cv1 = rows(rf"^\s*{SC}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+(\d+)\s+{NUM}", t101)
tables["coverage_seed42"] = pd.DataFrame(
    [(a, float(c), float(e), float(f), int(g), float(h)) for a, b, c, d, e, f, g, h in cv1],
    columns=["scenario", "coverage_all", "coverage_g3_pos", "coverage_g3_zero", "n_zero_grade", "avg_width"])

t94 = out_text("9.4")
dg = rows(r"Scenario ([ABC]): zero-grade students among the 8 largest errors = (\d)/8 \| q \(all\) = "
          r"(\d+\.\d+) \| q \(G3>0 only, for information\) = (\d+\.\d+)", t94)
tables["q_diagnostic_seed42"] = pd.DataFrame(
    [(a, int(b), float(c), float(d)) for a, b, c, d in dg],
    columns=["scenario", "zero_grade_in_top8_calib_errors", "q_all", "q_g3_positive_only"])

# ---- multi-seed summary (10.5): printed transposed ----------------------------------
t105 = out_text("10.5")
ms_rows = {}
for line in t105.splitlines():
    p = line.split()
    if len(p) == 4 and p[0] != "scenario":
        ms_rows[p[0]] = [float(x) for x in p[1:]]
ms = pd.DataFrame(ms_rows, index=["A", "B", "C"])  # rows = scenarios, columns = metrics
ms.index.name = "scenario"
tables["multiseed_summary"] = ms.reset_index()

# ---- permutation importance top-5 (11.3) --------------------------------------------
t113 = out_text("11.3")
perm = []
for sc, block in re.findall(r"^([ABC])\s+feature\s+mae_increase\n((?:\s*\w+\s+-?\d+\.\d+\s*\n)+)", t113, re.M):
    for rank, (feat, val) in enumerate(re.findall(r"(\w+)\s+(-?\d+\.\d+)", block), 1):
        perm.append((sc, rank, feat, float(val)))
tables["permutation_importance_top5_seed42"] = pd.DataFrame(
    perm, columns=["scenario", "rank", "feature", "mae_increase_calibration_set"])

# ---- error analysis (11.4) ----------------------------------------------------------
t114 = out_text("11.4")
ea = rows(rf"^\s*{SC}\s+{NUM}\s+{NUM}\s+{NUM}\s+(\d+)\s+(\d+)\s*$", t114)
tables["error_analysis_seed42"] = pd.DataFrame(
    [(a, float(b), float(c), float(d), int(e), int(f)) for a, b, c, d, e, f in ea],
    columns=["scenario", "mae_all", "mae_g3_pos", "mae_g3_zero", "n_g3_pos", "n_g3_zero"])
wp = rows(rf"^\s*{SC}\s+(\d+)\s+(\d+)\s+{NUM}\s+{NUM}\s*$", t114.split("Largest test errors:")[1])
tables["worst_predictions_seed42"] = pd.DataFrame(
    [(a, int(b), int(c), float(d), float(e)) for a, b, c, d, e in wp],
    columns=["scenario", "row_id", "y_true", "y_pred", "abs_err"])

# ---- library versions (2 and 51) ----------------------------------------------------
t2, t51 = out_text("0.3"), out_text("12.2")
ver = [(k.strip(), v.strip()) for k, v in re.findall(r"^(Python|pandas|numpy|scikit-learn|seaborn)\s*: (\S+)", t2, re.M)]
ver += [(k, v) for k, v in re.findall(r"^(matplotlib|joblib)==(\S+)$", t51, re.M)]
tables["library_versions"] = pd.DataFrame(ver, columns=["package", "version"])

# ---- write --------------------------------------------------------------------------
RES.mkdir(exist_ok=True)
for name, df in tables.items():
    df.to_csv(RES / f"{name}.csv", index=False)
    print(f"wrote results/{name}.csv  ({len(df)} rows)")

# ---- original figures embedded in the notebook outputs ------------------------------
FIGS = {  # old cell id -> file name
    "2.1": "g3_distribution.png", "2.3": "correlation_matrix.png", "2.4": "boxplots_g3.png",
    "10.2": "conformal_intervals_seed42.png", "11.1": "actual_vs_predicted_seed42.png",
    "11.2": "residuals_seed42.png", "11.3": "permutation_importance_seed42.png",
}
FIG.mkdir(exist_ok=True)
for old, fname in FIGS.items():
    for o in CELLS[old]["outputs"]:
        if o["output_type"] == "display_data" and "image/png" in o.get("data", {}):
            (FIG / fname).write_bytes(base64.b64decode(o["data"]["image/png"]))
            print(f"wrote figures/{fname}")
