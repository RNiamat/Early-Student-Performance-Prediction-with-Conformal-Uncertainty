"""Build README.md from scripts/readme_template.md and results/*.csv.

Every number in the README is read from results/*.csv (tables and inline values), none is typed by hand.
Usage (from the repo root):
    python scripts/build_readme.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SC = ["A", "B", "C"]


def rd(name):
    return pd.read_csv(RES / f"{name}.csv")


def md_table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


def pct(x, d=1):
    return f"{x * 100:.{d}f}"


ds = rd("data_summary").set_index("metric")["value"]
split = rd("split_summary")
ms = rd("multiseed_summary").set_index("scenario").loc[SC]
t42 = rd("test_results_seed42")
boot = rd("bootstrap_ci_seed42").set_index("scenario").loc[SC]
cov42 = rd("coverage_seed42").set_index("scenario").loc[SC]
qd = rd("q_diagnostic_seed42").set_index("scenario").loc[SC]
qc = rd("conformal_q_seed42").set_index("scenario").loc[SC]
cv = rd("cv_results")
tun = rd("tuning_results")
err = rd("error_analysis_seed42").set_index("scenario").loc[SC]
perm = rd("permutation_importance_top5_seed42")

n_rows = int(float(ds["n_rows"]))
n_test = int(split.loc[split.set == "test", "rows"].iloc[0])
n_zero_test = int(cov42["n_zero_grade"].iloc[0])
n_pos_test = n_test - n_zero_test
selected = t42[t42.selected == "yes"].set_index("scenario").loc[SC]
assert (selected["model"] == "RandomForest").all()

# ------------------------------------------------------------------ inline values
perm_first = lambda sc, r: perm[(perm.scenario == sc) & (perm["rank"] == r)].iloc[0]
V = {
    "n_rows": n_rows, "n_cols": int(float(ds["n_columns"])), "n_text": int(float(ds["n_text_columns"])),
    "n_num": int(float(ds["n_numeric_columns"])), "miss": int(float(ds["missing_values"])),
    "dup": int(float(ds["duplicate_rows"])),
    "g3_mean": f"{float(ds['g3_mean']):.2f}", "g3_std": f"{float(ds['g3_std']):.2f}",
    "g3_min": f"{float(ds['g3_min']):.0f}", "g3_max": f"{float(ds['g3_max']):.0f}",
    "n_zero": int(float(ds["n_zero_grade"])),
    "zero_pct": f"{100 * float(ds['n_zero_grade']) / float(ds['n_zero_grade_total']):.1f}",
    "abs0_pos": int(float(ds["absences_zero_among_positive"])), "abs0_pos_n": int(float(ds["n_positive_for_absences"])),
    "corr_abs_all": f"{float(ds['corr_absences_g3_all']):.2f}",
    "corr_abs_pos": f"{float(ds['corr_absences_g3_positive_only']):.2f}",
    "n_test": n_test, "n_zero_test": n_zero_test,
    "base20": f"{ms['baseline_mae'].iloc[0]:.2f}",
    "A_imp20": f"{ms.loc['A', 'improve_mean']:.2f}", "A_r2min": f"{ms.loc['A', 'r2_min']:.2f}",
    "A_pos": int(round(ms.loc["A", "improve_pos_share"] * 20)),
    "A_imp42": f"{boot.loc['A', 'improvement']:.2f}",
    "A_ci_lo": f"{boot.loc['A', 'improvement_ci_low']:.2f}", "A_ci_hi": f"{boot.loc['A', 'improvement_ci_high']:.2f}",
    "A_r242": f"{selected.loc['A', 'r2']:.2f}",
    "perm_failures": f"{perm_first('A', 1)['mae_increase_calibration_set']:.2f}",
    "perm_A2f": f"`{perm_first('A', 2)['feature']}`", "perm_A2v": f"{perm_first('A', 2)['mae_increase_calibration_set']:.2f}",
    "perm_G1": f"+{perm_first('B', 1)['mae_increase_calibration_set']:.2f}",
    "perm_G2": f"+{perm_first('C', 1)['mae_increase_calibration_set']:.2f}",
    "cov_lowest": pct(ms["cov_min"].min()),
    "pcov_min": pct(ms["cov_pos_mean"].min()), "pcov_max": pct(ms["cov_pos_mean"].max()),
    "zcov_min": pct(ms["cov_zero_mean"].min()), "zcov_max": pct(ms["cov_zero_mean"].max()),
    "A_wpct": f"{ms.loc['A', 'width_mean'] / 20 * 100:.0f}",
    "B_cov42": pct(cov42.loc["B", "coverage_all"]), "B_covmin": pct(ms.loc["B", "cov_min"]),
    "B_ridge42": f"{t42[(t42.scenario == 'B') & (t42.model == 'Ridge')]['mae'].iloc[0]:.2f}",
    "B_rf42": f"{selected.loc['B', 'mae']:.2f}",
}
assert perm_first("B", 1)["feature"] == "G1" and perm_first("C", 1)["feature"] == "G2"
assert perm_first("A", 1)["feature"] == "failures"
for s in SC:
    V[f"{s}_mae20"] = f"{ms.loc[s, 'mae_mean']:.2f}"
    V[f"{s}_r22"] = f"{ms.loc[s, 'r2_mean']:.2f}"
    V[f"{s}_cov20"] = pct(ms.loc[s, "cov_mean"])
    V[f"{s}_covsd"] = pct(ms.loc[s, "cov_std"])
    V[f"{s}_wid20"] = f"{ms.loc[s, 'width_mean']:.1f}"
    V[f"z42_{s}"] = f"{round(cov42.loc[s, 'coverage_g3_zero'] * n_zero_test)} of {n_zero_test}"
    V[f"top8_{s}"] = int(qd.loc[s, "zero_grade_in_top8_calib_errors"])
    V[f"qpos_{s}"] = f"{qd.loc[s, 'q_g3_positive_only']:.2f}"
    V[f"qall_{s}"] = f"{qd.loc[s, 'q_all']:.2f}"

# ------------------------------------------------------------------ tables
T = {}
T["split"] = md_table(
    ["Set", "Rows", "Share %", "G3 = 0 (count)", "G3 = 0 (share %)", "G3 mean", "G3 SD"],
    [[r.set, r.rows, f"{r.share_pct:.1f}", r.g3_zero_count, f"{r.g3_zero_share_pct:.1f}", f"{r.g3_mean:.2f}", f"{r.g3_std:.2f}"]
     for r in split.itertuples()])

err_rows = [[s, "RandomForest", f"{ms.loc[s, 'mae_mean']:.2f} ± {ms.loc[s, 'mae_std']:.2f}", f"{ms.loc[s, 'baseline_mae']:.2f}",
             f"{ms.loc[s, 'improve_mean']:.3f}", f"{int(round(ms.loc[s, 'improve_pos_share'] * 20))} / 20",
             f"{ms.loc[s, 'r2_mean']:.2f} (min {ms.loc[s, 'r2_min']:.2f})"] for s in SC]
conf_rows = [[s, f"{pct(ms.loc[s, 'cov_mean'])} ± {pct(ms.loc[s, 'cov_std'])} ({pct(ms.loc[s, 'cov_min'])}-{pct(ms.loc[s, 'cov_max'])})",
              pct(ms.loc[s, "cov_pos_mean"]), pct(ms.loc[s, "cov_zero_mean"]), f"{ms.loc[s, 'width_mean']:.1f}"] for s in SC]
T["multiseed"] = (
    "**Prediction error**\n\n"
    + md_table(["Scenario", "Model", "Test MAE (mean ± SD)", "Mean-baseline MAE", "Mean improvement", "Splits better than baseline",
                "Test R² mean (min)"], err_rows)
    + "\n\n**Conformal intervals (target coverage 90%)**\n\n"
    + md_table(["Scenario", "Coverage % (mean ± SD, min-max)", "Coverage % G3 > 0", "Coverage % G3 = 0", "Mean width"], conf_rows))

T["test42"] = md_table(
    ["Scenario", "Model", "Selected", "MAE", "RMSE", "R²"],
    [[r.scenario, r.model, "yes" if r.selected == "yes" else "", f"{r.mae:.3f}", f"{r.rmse:.3f}", f"{r.r2:.3f}"] for r in t42.itertuples()])

T["boot"] = md_table(
    ["Scenario", "Selected model", "Test MAE [95% CI]", "Baseline MAE", "Improvement [95% CI]"],
    [[s, "RandomForest", f"{boot.loc[s, 'test_mae']:.2f} [{boot.loc[s, 'mae_ci_low']:.2f}, {boot.loc[s, 'mae_ci_high']:.2f}]",
      f"{boot.loc[s, 'baseline_mae']:.2f}",
      f"{boot.loc[s, 'improvement']:.2f} [{boot.loc[s, 'improvement_ci_low']:.2f}, {boot.loc[s, 'improvement_ci_high']:.2f}]"] for s in SC])

cov_rows = []
for s in SC:
    k_all = round(cov42.loc[s, "coverage_all"] * n_test)
    k_pos = round(cov42.loc[s, "coverage_g3_pos"] * n_pos_test)
    k_zero = round(cov42.loc[s, "coverage_g3_zero"] * n_zero_test)
    cov_rows.append([s, f"{qc.loc[s, 'q']:.2f}", f"{qc.loc[s, 'quantile_level']:.3f}",
                     f"{pct(cov42.loc[s, 'coverage_all'])} ({k_all}/{n_test})",
                     f"{pct(cov42.loc[s, 'coverage_g3_pos'])} ({k_pos}/{n_pos_test})",
                     f"{k_zero}/{n_zero_test}", f"{qc.loc[s, 'avg_width_clipped']:.2f}"])
T["cov42"] = md_table(["Scenario", "q", "Quantile level", "Coverage % all", "Coverage % G3 > 0", "G3 = 0 inside", "Mean width (clipped)"], cov_rows) \
    + "\n\nCounts are derived from the printed coverage fractions."

T["cv"] = md_table(
    ["Scenario", "Model", "MAE", "RMSE", "R²"],
    [[r.scenario, r.model, f"{r.mae_mean:.2f} ± {r.mae_sd:.2f}", f"{r.rmse_mean:.2f} ± {r.rmse_sd:.2f}", f"{r.r2_mean:.2f} ± {r.r2_sd:.2f}"]
     for r in cv.itertuples()]) + "\n\nMean ± SD over 5 folds of the training set (default settings)."

T["tuning"] = md_table(
    ["Scenario", "Model", "Default CV MAE", "Tuned CV MAE", "Best parameters"],
    [[r.scenario, r.model, f"{r.default_cv_mae:.3f}", f"{r.tuned_cv_mae:.3f}", f"`{r.best_params}`"] for r in tun.itertuples()]) \
    + "\n\nTuned scores are the best of many settings and therefore optimistic. Several best values sit on the edge of the grid; the grid was not extended."

err_rows = []
for s in SC:
    r = err.loc[s]
    share = r.n_g3_zero * r.mae_g3_zero / (r.n_g3_pos * r.mae_g3_pos + r.n_g3_zero * r.mae_g3_zero) * 100
    err_rows.append([s, f"{r.mae_all:.2f}", f"{r.mae_g3_pos:.2f} (n = {int(r.n_g3_pos)})", f"{r.mae_g3_zero:.2f} (n = {int(r.n_g3_zero)})",
                     f"about {share:.0f}%"])
T["err"] = md_table(["Scenario", "MAE all", "MAE G3 > 0", "MAE G3 = 0", "Share of total absolute error from G3 = 0"], err_rows) \
    + "\n\nThe share is derived from the printed (rounded) MAEs."

perm_rows = []
for rk in range(1, 6):
    perm_rows.append([rk] + [f"`{perm_first(s, rk)['feature']}` ({perm_first(s, rk)['mae_increase_calibration_set']:.3f})" for s in SC])
T["perm"] = md_table(["Rank", "A", "B", "C"], perm_rows)

# ------------------------------------------------------------------ fill template
text = (ROOT / "scripts" / "readme_template.md").read_text(encoding="utf-8")
text = re.sub(r"\{\{table:(\w+)\}\}", lambda m: T[m.group(1)], text)
text = re.sub(r"\{\{(\w+)\}\}", lambda m: str(V[m.group(1)]), text)
left = re.findall(r"\{\{.*?\}\}", text)
assert not left, f"unfilled placeholders: {left}"
(ROOT / "README.md").write_text(text, encoding="utf-8")
print("README.md written,", len(text.splitlines()), "lines")
