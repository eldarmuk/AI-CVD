# Historical training and evaluation pipeline source

These retained source files document the earlier numbered research workflow. They
are **source archives**, outside the installed package and current public CLI.
They retain historical `src.*` imports, artifact layouts, optional dependencies and
task assumptions. Do not treat them as fresh-checkout commands or current validated
clinical pipelines. The complete original checkout remains preserved internally.

Use the maintained [raw-to-results pipeline](../../docs/PIPELINE.md) for executable
public examples. The [history map](../../docs/architecture/pipeline-history.md)
explains how processing was repaired and how later studies differ.

| Source | Historical purpose |
| --- | --- |
| [02_train_vae.py](02_train_vae.py) | Normalization, LSTM-VAE fitting and validation |
| [03_evaluate.py](03_evaluate.py), [03b_evaluate_auprc.py](03b_evaluate_auprc.py) | Reconstruction scores, ranking metrics and reports |
| [04_analyze_results.py](04_analyze_results.py) | Exploratory feature/window/subject analysis |
| [05_train_grid_search.py](05_train_grid_search.py) | Earlier model search |
| [06_flatten_tabular_features.py](06_flatten_tabular_features.py) | Sequence-to-tabular summaries |
| [07_run_classical_baselines.py](07_run_classical_baselines.py) | Classical anomaly and supervised baselines |
| [08_benchmark_report.py](08_benchmark_report.py) | Historical comparison reports |
| [09_train_supervised_sequence.py](09_train_supervised_sequence.py) | Earlier supervised sequence model |
| [10_analyze_shap_rf.py](10_analyze_shap_rf.py) | Exploratory attribution and operating-point analysis |
| [11_train_cgta_net.py](11_train_cgta_net.py) | Earlier CGTA model |
| [12_train_hybrid_cgta.py](12_train_hybrid_cgta.py) | Earlier hybrid architecture |
| [13_extract_cgta_embeddings.py](13_extract_cgta_embeddings.py) | Embedding-generation source; embeddings are not included |
| [14_train_deep_tree_fusion.py](14_train_deep_tree_fusion.py) | Earlier deep/tree fusion |
| [run_comparisons.py](run_comparisons.py) | Feasibility-era model comparisons |

Formatting and unused-import/local cleanup were applied to release copies only.
Archived runners are syntax/lint checked, not executed against clinical data. Source
presence is not a claim of runtime reproducibility, clinical validity or absence of
historical bugs. Some analyses choose operating points on the evaluated test set;
those must not be reused as unbiased prospective performance estimates.

No historical outputs, fitted weights, predictions, raw source connectors, private
note classification or provider database definitions accompany this archive.
