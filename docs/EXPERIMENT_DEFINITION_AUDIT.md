# Experiment-definition repair inventory

This is a source-code audit, not a clinical result. Canonical definitions are in
`configs/tasks/level3_4h.toml`; they are intentionally incompatible with v1 arrays.

| Definition or path | Historical conflict | v2 resolution |
|---|---|---|
| Feature dictionary | 96 five-minute steps described as 24h; 24h labels | 8h lookback, 4h forecast |
| `00_build_multimodal_features_duckdb.py` | 4h labels, first-only alert compression, availability at bucket start | Canonical CLI wrapper; shared feature implementation and explicit episodes |
| `00a_filter_elite_cohort.py` | Entire-history density using first/last readings | Fixed run-in from externally declared enrollment |
| `01_generate_dataset.py` | L2/3 positives, pure-normal future labels in input eligibility, fixed event-patient reservations, balanced test | Patient hash split before windows; independent prediction grid; primary L3 complete stream |
| `01b_extract_mixed_train.py` | Same future-label sampler, same-patient balanced historical cache | Optional separate train/validation development manifest |
| `process_data.py` | Discarded later burst alerts; clipped physiology; undated clinical history | `process_alerts` builds v2 tables; old ETL entry point retired; primary cleaning reads raw values |
| `utils.get_feature_columns` | Accept everything except a short denylist | Exact ordered v2 allowlist; reject incomplete/legacy schemas |
| `02_train_vae.py`, `03_evaluate.py`, `03b_evaluate_auprc.py` | 96 steps plus stale architecture/real-world labels | Explicitly retired historical commands, no v2 training/evaluation claim |
| `04_analyze_results.py`, `05_train_grid_search.py` | Fixed old feature positions/sequence constants | Explicitly historical; migration deferred |
| `06_flatten_tabular_features.py` | L2/3 copied into `*_l3` files; active-window filtering changes population | Explicitly historical; v2 sample identity required |
| `07_run_classical_baselines.py`, `08_benchmark_report.py` | Clock-ablated comparisons, stale result coalescing | Explicitly historical; no v2 primary use |
| `09` through `14` sequence, SHAP, hybrid and fusion scripts | Different endpoints/cohorts, length-only joins, test-derived operating points | Direct execution blocked pending a separate training/evaluation repair |
| `run_comparisons.py` | Duplicated models and assumed deployment prevalence | Explicitly historical |
| `src/archive/*.py` | Older 24h/96-step experiments | Historical headers and execution guards |
| `src/archive/legacy_v1/` | Original four data scripts and old sampler tests | Preserved for audit; not primary executables |
| Existing notebooks, reports and models | Outputs reflect earlier experiments and may contain patient information | Historical private artifacts, not updated/recomputed; not primary evidence |

The canonical data CLI does not read old `.npy`, `.npz`, normalizers, patient
splits, metric JSON or elite-cohort CSV files. New run metadata is required and
content-checked by every new exporter. Numerical constants in retired code are
left visible for historical audit rather than misleadingly edited to look current.

The old regression tests validated isolated target rows on a 15-minute grid.
The new test constructs real four-hour lookahead semantics and requires all
48 positive prediction times. Retaining only the first positive time fails it.

Source-schema inspection can establish that monitoring/ascertainment/history
availability fields are absent; it cannot establish that patients were enrolled
through the last measurement or that baseline conditions were known earlier.
Data-derived step audit statistics are kept under ignored `data/private_audit/`.
