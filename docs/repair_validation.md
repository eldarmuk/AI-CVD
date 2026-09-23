# Problems 1–6 implementation and validation

## Implemented scope

| Problem | Implementation |
|---|---|
| 1. Freeze task | `configs/tasks/level3_4h.toml`, `src/ai_cvd/task.py`; 5-minute grid, 96 rows, 480-minute lookback, 240-minute Level-3 horizon. Duration/schema assertions, content fingerprints, explicit secondary endpoint, retired legacy executables. |
| 2. Episodes | `episodes.py`, `components/process_data.py`; inclusive chained 10-minute episodes preserve every member, first/max/final severity and escalation timing. Targets use first qualifying recorded severity, not low-severity episode start. |
| 3. Causal features | `features.py`; bucket closure, HR/SBP, paired PP, inclusive validity bounds, valid-only recency, explicit masks, conservative declared counter differencing/reset policy, dated/excluded histories. |
| 4. Cohort/follow-up | `dataset.py`, `cli.py`; fixed enrollment run-in, external continuous coverage, censoring, auditable cohort flow. No first/last-measurement inference or lifetime-density eligibility. |
| 5. Windows/identity | Separate complete stream, balanced development and retrospective training manifests; patient split before window selection; task/endpoint/source-manifest-aware joins; immutable patient/time IDs; checksummed NPZ shards and per-row input hashes. |
| 6. Regression fixture | `synthetic.py`, `tests/test_scientific_pipeline.py`, `tests/test_pipeline_integrity.py`; wholly fictional records, no private-data requirement. |

The primary dictionary's explicit feature order is checked against `FEATURE_NAMES`.
Historical scripts and results remain historical; no model fitting, architecture
changes, threshold selection, or Problems 7+ work is included.

## Verification

The full synthetic suite passed **45 tests**, including SQLite/DuckDB source and
output parity, source immutability, 48 positive prediction times for one event,
timing boundaries, censoring, split isolation, masks, invalid observations,
duplicate pairs, daily counter resets, schema consistency, source declarations,
and mismatched/permuted artifacts and prediction joins. The initial adapter run
could not read the locally installed dependency under the sandbox; the complete
rerun with access to that dependency passed without skipped tests.

The final CLI smoke run passed synthetic build, development extraction and verified
array loading for all partitions: 1,297 retrospective training samples, 529
validation-stream samples, and 1,058 test-stream samples. These counts describe
fictional data only. The fixture explicitly exercises all three hash partitions.

Python source parsing and historical AST comparisons passed for 31 retained
implementations. All 32 retired command guards and four canonical wrappers passed.
Inherited trailing whitespace in one archived file was removed without changing
its Python semantics. Git whitespace checks passed after cleanup.

Reproduce with Python 3.11+, `requirements-data.txt`, and a provisioned DuckDB
SQLite extension:

```powershell
python -m unittest discover -s tests -v
```

## Remaining source limitations

These are unresolved scientific prerequisites, not measurements inferred by code:

- Enrollment and continuous measurement/outcome ascertainment intervals need
  independent evidence. Last measurement is not discharge or outcome coverage.
- Cumulative daily-reset Steps have strong empirical support from the local audit,
  but device semantics and reset timezone are not authoritatively documented.
- Measurement delivery latency and naive source timezone require declarations.
- Alert notes lack separate severity-recording timestamps. Equating recorded
  severity time with alert creation must be evidenced or explicitly treated as a
  retrospective timing assumption; the classifier remains an unadjudicated proxy.
- Undated disease/medication histories cannot be used causally without provenance;
  the safe available policy is `exclude`.
- Historical test-set exposure is not undone by this repair. New validation design
  and clinical adjudication remain later scientific work.

The implementation is regression-tested, but Problems 1–6 cannot be declared
unconditionally scientifically complete for the private dataset until the coverage
and timing declarations are resolved. The pipeline rejects missing declarations.
Private production-scale throughput has not been benchmarked.

## Private extraction after declarations are resolved

Prepare `data/private/source-contract.toml` from the example and a coverage CSV
with the columns documented in the study protocol. Keep undated history excluded.
Use a frozen raw database, an owner-assigned immutable snapshot ID, and new output
directories. The following commands generate data only:

```powershell
python -m src.ai_cvd.cli build --raw-db db/hrp_data.db --source-contract data/private/source-contract.toml --coverage data/private/coverage.csv --output runs/level3-4h-v2 --source-snapshot-id (Read-Host "Immutable source snapshot ID")
python -m src.ai_cvd.cli sequences --run runs/level3-4h-v2 --output runs/level3-4h-v2-train --split train --kind unsupervised_training
python -m src.ai_cvd.cli sequences --run runs/level3-4h-v2 --output runs/level3-4h-v2-validation --split validation
python -m src.ai_cvd.cli sequences --run runs/level3-4h-v2 --output runs/level3-4h-v2-test --split test
```

Use `arrays.verified_shards` when loading these arrays. Preserve source-manifest
checksums alongside all future predictions. No real source records, manifests,
arrays, audit statistics, checkpoints or clinical notes belong in Git. Existing
tracked private artifacts/history require a separate release review; ignore rules
do not retroactively remove them.
