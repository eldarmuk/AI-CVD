# AI-CVD — causal data pipeline, task v2

The primary task is **Level 3 escalation within four hours**, using **96 five-minute
buckets (eight hours)** of preceding observations. The authoritative configuration
is [configs/tasks/level3_4h.toml](configs/tasks/level3_4h.toml). Read the
[study protocol](docs/study_protocol.md) and [feature dictionary](docs/FEATURE_DICTIONARY.md).

Historical results are not evidence for this repaired task. Training and evaluation
scripts are explicitly retired pending migration; this repair does not change model
architecture, train models, or select thresholds.

## Quick synthetic verification

Python 3.11+ is required. The core tests and feature/manifest generation use the
standard library; sequence-export tests require NumPy. Use `requirements-data.txt`
for data work. The broader historical requirements are not a reproducible training
environment.

```powershell
python -m unittest discover -s tests -v
python -m src.ai_cvd.cli synthetic --output runs/synthetic-v2
python -m src.ai_cvd.cli development --run runs/synthetic-v2 --output runs/synthetic-development-v2 --per-class 20
python -m src.ai_cvd.cli sequences --run runs/synthetic-v2 --output runs/synthetic-test-arrays-v2 --split test
```

Output directories must be new. All generated outputs are ignored by default,
including synthetic ones, so no private metadata accidentally enters a release.

## Private extraction — explicit source declarations required

Do not infer coverage from the first/last measurement. Prepare a private coverage
CSV using externally established continuous monitoring and outcome intervals:

```text
senior_id,enrollment_time,measurement_coverage_end,outcome_coverage_start,outcome_coverage_end
```

Use timezone-aware dates. Copy `configs/source_contract.example.toml` to a private
location and resolve every unknown declaration. Either exclude undated clinical
history or provide dated numeric snapshots as JSONL records with `senior_id`,
`available_at` and `values`. There is no automatic join to undated disease tables.

```powershell
python -m src.ai_cvd.source_audit --raw-db db/hrp_data.db --output data/private_audit/source-audit.json
python -m src.ai_cvd.cli build --raw-db db/hrp_data.db --source-contract data/private/source-contract.toml --coverage data/private/coverage.csv --output runs/level3-4h-v2 --source-snapshot-id OWNER_ASSIGNED_IMMUTABLE_SNAPSHOT
```

The default SQLite backend is read-only. `--backend duckdb` uses a provisioned
DuckDB SQLite extension and the same feature/episode logic; it does not install
extensions or contact a service automatically. Do not run extraction against a
changing database snapshot. Do not substitute `hrp_processed.db`: its historical
clipping/debounce has already discarded information.

Adapter parity tests require DuckDB's SQLite extension to be provisioned locally
(`python -c "import duckdb; duckdb.connect().execute('INSTALL sqlite')"`). If using
a custom extension directory, set `AI_CVD_TEST_EXTENSION_DIRECTORY` for tests and
`duckdb_extension_directory` in the private source contract for extraction.
Read exported arrays through `src.ai_cvd.arrays.verified_shards`, preserving their
sample IDs; never align predictions by row position alone.

The secondary endpoint requires explicit `--endpoint level2_or_level3` and a
different output directory. Optional retrospective VAE training policies are
`event_free_windows` (default) and `never_event_patients`; neither changes the
natural-prevalence stream. Model fitting is a future task.

## Privacy and release

Never commit source records, source audit results, coverage/history declarations,
sample/prediction manifests, arrays, embeddings, weights, notes or patient plots.
Hash IDs are pseudonyms, not anonymization. This repository historically tracked
patient-derived artifacts; ignore rules do not remove existing files or history.
No public release or history rewrite is authorized by this data repair alone.

Public reproducibility uses only the handwritten fictional generator and tests.
See [definition audit](docs/EXPERIMENT_DEFINITION_AUDIT.md) for the historical-code
inventory and remaining assumptions.
