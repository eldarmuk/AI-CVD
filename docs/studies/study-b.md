# Study B — SOS-conditioned recorded-outcome prediction

**Status: completed study; manuscript in preparation.**

Study B starts from an SOS episode and uses strictly earlier wearable history to
predict recorded telecare outcomes. It supports a research prioritization question,
not continuous detection of a future alarm. L3 is a recorded outcome category;
independent clinical adjudication and deployment usefulness are not established.

| Held-out aggregate | Value |
| --- | ---: |
| AUROC | 0.67102 |
| Average precision | 0.05391 |
| Sensitivity at the frozen operating point | 85.19% |
| Episodes prioritized | 56.63% |
| Test L3 episodes | 27 |

The small positive-class count limits precision. High sensitivity comes with a large
review workload. These supplied frozen aggregates are documentation, not values
recomputed by the synthetic demo. No confidence interval is invented here.

[Episode summaries](../../src/ai_cvd/features/episode.py),
[temporal preprocessing](../../src/ai_cvd/features/temporal.py),
[GRU-D](../../src/ai_cvd/models/grud.py) and
[mTAN](../../src/ai_cvd/models/mtan.py) are public-release copies of selected internal
implementations. The mTAN model is a study-specific adaptation of an established
method, not a novel attention mechanism or a benchmark reimplementation.

The [complete processing pipeline](../PIPELINE.md) now builds these inputs from raw
fictional telemetry through DuckDB and Parquet. The [stage map](../architecture/pipeline-history.md)
connects processing, model selection, calibration and evaluation to the preserved
internal research workflow.

The [model demo](../../examples/synthetic/README.md) uses explicit fictional outcome codes,
not private note text or institution-specific labeling rules. Its logistic baseline
is fitted only on fictional training subjects; neural networks remain untrained.
Thresholds are chosen on fictional validation subjects and applied to separate
fictional test subjects. All toy outputs are explicitly marked synthetic.
