# Pipeline evolution and historical stages

The research has a processing lineage, not just a collection of models. The earlier
release candidate incorrectly omitted too much pipeline source. This revision
restores maintained algorithms and historical training/evaluation source while
continuing to exclude generated artifacts and institution-specific connectors.

| Original stage | Present location or boundary |
| --- | --- |
| components/load_data.py and database.py | Original source-specific spreadsheet/SQLite ingestion remains in the research checkout; [storage.py](../../src/ai_cvd/pipeline/storage.py) provides the explicit public CSV/DuckDB boundary |
| components/process_data.py | Historical rounding, clipping and source-table orchestration remain preserved; repaired validation is in [features.py](../../src/ai_cvd/pipeline/features.py) |
| pipelines/00_build_multimodal_features_duckdb.py | Repaired entry point delegated to the canonical builder; [runner.py](../../src/ai_cvd/pipeline/runner.py) and retained feature builders expose the public path |
| pipelines/00a_filter_elite_cohort.py | Earlier whole-record cohort filtering was superseded; retained run-in/coverage eligibility is in [dataset.py](../../src/ai_cvd/pipeline/dataset.py) and [index.py](../../src/ai_cvd/pipeline/index.py) |
| pipelines/01_generate_dataset.py and 01b_extract_mixed_train.py | Retained subject splits, windows, retrospective training selection and lazy batches in dataset.py / runner.py |
| ai_cvd/features.py | [Retained causal feature implementation](../../src/ai_cvd/pipeline/features.py) |
| ai_cvd/fast_features.py | [Retained vectorized equations](../../src/ai_cvd/pipeline/vector.py); private SQL reader replaced by canonical records |
| ai_cvd/compact.py | [Retained window-index algorithm](../../src/ai_cvd/pipeline/index.py); public orchestration and compact arrays in runner.py |
| pipelines/02 through 14 and run_comparisons.py | [Historical source archive](../../legacy/pipelines/README.md) |
| architecture_study and final_study | [Study A](../studies/study-a.md): model definitions and repaired processing interfaces are public; frozen training/resume identities remain preserved internally |
| study_b_b1 through study_b_b3 | [Study B feature/model interfaces](../studies/study-b.md) and [processed-data bridge](../../src/ai_cvd/pipeline/study_b.py) |
| study_b_b4, study_b_b5, study_b_b6 | Internal selection/calibration, held-out evaluation and final reporting; frozen weights/predictions/provenance remain private. Approved aggregates are documented on the Study B page |

## Scientifically meaningful changes

The older processing code rounded/clipped some measurements and mixed earlier cohort
and task assumptions. The repaired canonical implementation masks invalid values,
preserves paired-pressure validity, requires explicit availability/coverage semantics,
and separates candidate eligibility from target labeling. Those changes must be
described in a thesis; they are not cosmetic refactoring.

Continuous Study A windows and SOS-conditioned Study B episodes are different dataset
products. An alarm anchor, a retrospectively qualifying constituent and the time an
outcome became known are not interchangeable. The numeric-outcome public adapter
does not reimplement private note classification.

The archived 02–14 runners retain earlier task assumptions, model searches, reporting
logic and known limitations. In particular, archived test-set operating-point analyses
are descriptive historical analyses, not a valid replacement for validation-only
threshold selection. Historical 0.782 results remain exploratory and are not a
directly comparable improvement over the published or later tasks.

## Route to the current results

Raw source -> source-specific ingestion/normalization -> repaired causal grids ->
coverage and cohort gates -> task-specific windows/episodes -> training-only
preprocessing -> internal model selection -> frozen calibration/priority policy ->
held-out evaluation -> final aggregate reporting.

The public pipeline implements and tests the source-neutral processing and input
construction route and a fictional model/evaluation route. It does not silently
substitute fictional fitting for frozen B4/B5/B6 procedures or claim to regenerate
their clinical metrics. Exact internal executions remain bound to their preserved
code/configuration/data identities. No frozen original was moved or edited.
