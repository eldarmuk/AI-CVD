# Limitations

The public processing interface consumes local canonical batch exports. It is not a
live wearable gateway. Processing uses a single DuckDB writer and materializes each
subject's history; large-subject memory and real-source scaling need workload-specific
measurement. Its vector path supports immediate availability only. Historical source
runners are retained for research traceability but are not fresh-checkout interfaces.
See [pipeline limitations](PIPELINE.md#reliability-and-performance).

- Retrospective recorded telecare outcomes are not automatically independently
  adjudicated clinical events. Alarm initiation, outcome recording and clinical
  deterioration are different times and concepts.
- An SOS-conditioned population differs from a continuously monitored population.
  Cross-study AUROC rankings are invalid without matched tasks and evaluations.
- Study B has only 27 test L3 episodes. Sensitivity of 85.19% at 56.63% prioritized
  implies a substantial workload; neither precision nor operational usefulness
  should be inferred from sensitivity alone.
- Missingness, acquisition patterns, personal baselines and source-specific labeling
  can affect both performance and transfer to another service or population.
- There is no established prospective deployment readiness, clinical utility,
  independent external validation or validated cardiovascular detection claim.
- mTAN and GRU-D are established method families. Included implementations are
  study-specific adaptations; the repository makes no novel mTAN mechanism claim.
- Private data and fitted artifacts are unavailable publicly. Synthetic examples test
  software behavior, not clinical performance or full scientific reproducibility.
- Frozen internal results and historical corrections were not recomputed or altered.
  Earlier CGTA/fusion AUROC 0.782 belongs to a superseded exploratory task.
- Retained legacy code has interface smoke coverage, not a claim of comprehensive
  scientific or software validation. The AIME decoder still samples its latent
  variable on forward calls; evaluation-mode use alone does not make it deterministic.
- Public adapters and formatting differ from frozen internal source bytes. Their
  tested behavior and migration scope are described in the [code map](architecture/code-map.md).
- Software ownership, institutional permissions and third-party rights must be
  cleared before licensing and publication. No license is granted by this candidate.
