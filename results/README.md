# results/

Tables reconstructed by `scripts/extract_from_notebook.py` from the text printed in the saved notebook outputs
(the original CSV checkpoints were lost). Values keep the printed precision; nothing was recomputed.

| File | Source cell (notebook) |
|---|---|
| data_summary, g3_correlations | 08, 09, 12, 13, 14, 16 |
| split_summary | 21 |
| cv_results | 29 |
| tuning_results | 32 |
| test_results_seed42, bootstrap_ci_seed42 | 35, 36 |
| conformal_q_seed42, q_diagnostic_seed42, coverage_seed42 | 40, 41, 42 |
| multiseed_summary (20 splits, summary only) | 45 |
| permutation_importance_top5_seed42, error_analysis_seed42, worst_predictions_seed42 | 48, 49 |
| library_versions | 02, 51 |

Not recoverable: per-student predictions/intervals, per-split results of the 20 resplits, model files.
