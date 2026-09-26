# R0/R1 implementation foundation — study v1.0.0

Validated on 2026-09-26. The foundation is ready for a bounded **canonical-training-only**
development experiment. No private encoder/head fitting, architecture selection,
canonical validation scoring or canonical test scoring was performed. This is not
evidence that either architecture predicts the clinical endpoint successfully.

The authoritative scientific design remains [architecture_selection.md](architecture_selection.md).
The executable definition is [r0_r1_v1.toml](../configs/model_studies/r0_r1_v1.toml).
It binds the task identifier to `configs/tasks/level3_4h.toml` and the existing
`runs/level3-4h-attested-20260924T095744Z`. No canonical shards, labels,
normalization or source-pipeline files were changed. All generated study artifacts
are restricted to new directories under ignored `runs/model_studies/`.

## Implemented models and projection

`src/architecture_study/` is separate from `src/ai_cvd/` to preserve the data
pipeline fingerprint. Projection `primitive6_process25_v1` validates the original
59-column order before selecting:

- Six value/target channels: temperature, heartrate, SBP, DBP, saturation, valid
  Steps increments. Raw cumulative Steps, pulse pressure, shock index, rolling
  summaries and undated static histories never enter the model.
- Process context: six natural masks, six known-recency indicators, six
  `log1p(recency_minutes)` values, three existing UTC clock fields, Steps interval
  log-duration/known flag, Steps source-observation flag and reset flag. These 25
  channels contain acquisition information, not hidden physiological magnitudes.

**R0:** one width-32 GRU-D-style value encoder, positive learned decay, and
evidence-masked attention pooling. A fixed mean of the 25 process-context channels
is concatenated with the value summary and two evidence flags. This fixed summary
gives R0 the same permitted information and an explicit observation-context fallback;
it is not a second learned temporal branch. Its representation has 59 dimensions.

**R1:** the identical value encoder plus a width-16 process GRU and its own temporal
attention pool. Concatenate the two summaries and the same evidence flags: 50
dimensions. This comparison tests learned temporal process encoding versus R0's
fixed summary; it is not a capacity-matched proof of disentanglement.

The value encoder accepts values, visible masks, visible-age/known flags, and
natural mask/recency context. Wholly invisible rows only decay the recurrent state.
The cache contains only visible values; time since its last update uses closed
five-minute buckets within the current 96-row window. State resets at every window.
This is a documented GRU-D-style modification, not a faithful original GRU-D replica.

The two evidence flags are absence of any visible primitive value and fraction of
visible primitive cells. With no visible values, the physiological pool is exactly
zero; the process representation remains available. All-empty samples are retained
for risk prediction. A Steps-only window has activity evidence, not necessarily
direct vital-sign evidence; the generic primitive-evidence flag does not claim otherwise.

Both recipes use a width-32 tanh SSL decoder conditioned on pooled representation
and relative grid time, with six output channels and no input skip connection.
Decoder input widths differ because representation widths differ; the counts below
include this difference. Risk export discards the decoder, copies/freezes the
encoder, and fits only an intercept plus L2-regularized logistic coefficients.
Head embedding standardization is fitted using weighted **head-training** embeddings
and algebraically absorbed into the exported linear layer. L2 penalizes the
standardized coefficients, excluding the intercept.

| Parameters | R0 | R1 |
|---|---:|---:|
| SSL encoder + pooling | 7,527 | 9,608 |
| SSL decoder | 2,150 | 1,862 |
| Total SSL trainable | **9,677** | **11,470** |
| Exported frozen-encoder risk model | 7,587 | 9,659 |
| Trainable during head fitting | 60 | 51 |

R2, R3 and R4 are rejected by the model constructor. No cross-attention, variational
latent, suffix forecasting or gradual fine-tuning is implemented in this version.

## Masking, loss and identity safeguards

Natural validity `M`, artificial mask `A`, and visible validity `V=M & ~A` remain
separate. Corruption hides 20% of naturally observed cells within each window,
rounded to at least one target when at least two observations exist, while retaining
at least one visible observation. Empty/single-reading windows have no SSL targets.
Natural acquisition timing may remain visible; hidden value magnitudes cannot
enter the value cache or reset visible-value age. This is value denoising, not
simulation of acquisition dropout.

Huber loss uses standardized targets, averages over targets within each contributing
sample/channel pair, then averages those pair means. Indexed target selection avoids
`NaN * 0`. Naturally missing/visible unhidden cells are not reconstruction targets.
A batch without targets has finite zero loss and **no optimizer step**.

The loader verifies the existing independent run verification, task/schema,
training-export hash, canonical patient split and consumed shard hashes. Every
selected sample ID is recomputed from patient and prediction time. Windows reconstruct
exactly 96 preceding rows. Wrong manifest IDs, reordered ID-keyed predictions, duplicate
IDs and inconsistent episode/target intervals fail. There is no validation/test loader
option in the study runner. IDs and episode links are audit metadata, never neural inputs.

SSL/risk checkpoints have immutable metadata sidecars binding weight checksum,
recipe, full config fingerprint, scaler fingerprint, task/projection, fold scope,
canonical run and train-export fingerprints. Loading checks all identities and uses
PyTorch `weights_only=True`. A risk checkpoint contains no decoder.

## Bounded population and sampling

Three deterministic patient-grouped folds are made **inside canonical train only**,
stratified by training-patient eligible-event presence. At least three event patients
are required. Inner held patients may select SSL checkpoints and assess the fitted
head; they never enter SSL updates, scaler/head fitting or a reference-density fit.
Canonical validation/test patients cannot enter any of these operations.

Pretraining uses the training **stream**, ignoring window labels, rather than
retrospective event-free selection. Each fold has its own label-blind scaler fitted
on unique buckets covered by 4,096 sampled training-fold windows. Duplicate/overlapping
window rows are counted only once. Channels absent from this sampled scaler use
mean zero/scale one and are explicitly recorded; such absence needs review before
interpreting that channel's reconstruction. The canonical normalizer is preserved.

Sampling is patient-first, then uniform 8-hour UTC blocks, then uniform windows
within a block. A label-blind pool caps distinct windows per patient at 256.
Pool/draw conditional probabilities, IDs, block IDs and seeds are saved. Visits
sample the fixed pool without replacement within a cycle; a short remainder is
discarded before starting the next full visit. Short patient pools contribute all
available distinct windows, so a batch may contain fewer than 64 samples. This
does not exclude short patients from supervised assessment.

The default batch has four patients, up to 16 windows each. The cap is 2,000 draws
per fold/recipe, at most 128,000 window exposures, with fewer actual optimizer
updates if some draws have no targets. Six fits allow at most 768,000 exposures.
Exposure counts are not distinct patients, windows or clinical events.

R0/R1 reuse the same saved per-fold sampling plan, scaler and held checkpoint samples.
Dedicated corruption generators restart at the same seed for both recipes. The
runner hashes ordered sample IDs **and actual artificial-mask bytes** across SSL
draws and asserts identical R0/R1 hashes. Synthetic integration tests exercise this
check across all three folds. Identical seeds alone are not the pairing proof.

The head uses up to four windows per positive linked-episode-set stratum and one
negative window per patient, capped at 20,000 samples. Multiple alarms in a horizon
remain linked; windows are not counted as independent events. Known inclusion
probabilities provide inverse-probability weights for head fitting and inner-fold
development estimates. These sampled AP/log-loss estimates are **not** full-stream
operational metrics or proof of calibration. No final architecture decision, clinical
threshold, few-shot superiority claim or event-level sensitivity is emitted by this
foundation. Natural-stream operational evaluation and patient-clustered uncertainty
remain necessary under the approved plan before substantive selection claims.

## Validation performed

The final complete suite passed **74/74 tests, zero failures and zero skips**:
56 existing data-pipeline tests plus 18 architecture-foundation tests. This includes:

- M/A/V, hidden-cache and visible-age invariance, excluded-derived-channel perturbation,
  missing-value gradients, channel-balanced Huber and zero-target optimizer skipping.
- Empty/single-reading windows, zero Steps versus missingness, known-zero versus
  unknown recency, and all-masked pooling.
- Future-suffix mutation invariance at prediction cutoff, prefix completeness,
  canonical feature order, sample-ID alignment and corrupted shard rejection.
- Isolated patient folds, training-only authorization, label-blind sampling, short
  patient pools, independently checked unique-row scaling and episode links.
- Tiny optimization of both recipes, frozen-head parameter equality, deterministic
  repeated inference, identity-checked checkpoint round trips and rejected mismatches.
- A fully fictional three-fold/two-recipe runner test: two SSL draws per fit,
  small heads, matched corruption hashes and successful artifact generation.

The first full-suite attempt found the GPU environment's DuckDB 1.5.1 did not have
a matching cached SQLite extension. A repository-local ignored DuckDB 1.5.5 install
matched the existing extension. The final run included the SQLite/DuckDB parity test;
it was not skipped. Existing pandas downcasting deprecation warnings remain in the
unchanged canonical pipeline and did not cause failures.

Only four canonical **training** windows from two training patients were loaded for
the private smoke check, with no optimizer/head fitting and no window-label reads.
A CPU repeat of these same windows checked the finalized scaler/source binding and
timed shard access. No canonical validation/test outcome was inspected. The smoke
scalers are explicitly limited test artifacts, not usable study-fold scalers.

Ignored local evidence paths:

- `runs/model_studies/r0-r1-synthetic-cuda-20260926-v1/benchmark.json`
- `runs/model_studies/r0-r1-synthetic-cpu-20260926-v1/benchmark.json`
- `runs/model_studies/r0-r1-train-loader-20260926-v1/loader_smoke.json`
- `runs/model_studies/r0-r1-train-loader-cpu-20260926-v2/loader_smoke.json`
- `.tmp/r0-r1-full-tests-verified.log`

## Resource measurements and recommendation

Hardware: RTX 4060 Laptop GPU, 8 GiB; CPU runs used four PyTorch threads.
Environment: Python 3.11.14, PyTorch 2.5.1+cu121, NumPy 2.4.6, pandas 2.3.3.
Each synthetic case used two warm-up updates and five timed updates. All measured
loss sequences decreased and remained finite. This checks gradient flow, not convergence.

| Batch 64 measurement | R0 | R1 |
|---|---:|---:|
| GPU seconds/update | 0.596 | 0.540 |
| GPU windows/second | 107.4 | 118.6 |
| GPU peak allocated bytes | 86,828,032 | 104,975,872 |
| GPU peak reserved bytes | 106,954,752 | 106,954,752 |
| CPU seconds/update | 0.174 | 0.214 |
| CPU windows/second | 367.3 | 298.8 |
| CPU process peak resident bytes | 496,951,296 | 504,201,216 |

The canonical GPU loader/forward smoke (batch 4, no gradients) peaked at
34,852,352 allocated bytes for R0 and 42,554,880 for R1. Corresponding reserved
memory was 35,651,584 and 56,623,104 bytes. These are PyTorch allocator measurements,
not total driver/context or system GPU memory. GPU-process peak host RSS during the
synthetic benchmark reached about 1.39 GB. Batch 64 is demonstrated feasible on the
8 GiB GPU; larger batches were not tested or selected.

The tiny recurrent loop is faster on CPU in these measurements, plausibly because
CUDA launch/synchronization overhead dominates small operations. **Recommend CPU,
four threads, batch cap 64**, keeping the paired sampling budget fixed. CUDA remains
a validated alternative. Short benchmarks do not establish a reliable speed ranking
between recipes, and private I/O can dominate either device.

For 3 folds × 2 recipes × 2,000 updates, measured model-compute extrapolation is
about **39 CPU minutes or 114 GPU minutes**. It excludes checkpoint/head/scaler work,
plan generation and cold shard access. Two training-shard hash/decode smoke timings
were approximately 18–22 ms; at 48,000 worst-case patient visits during SSL, that
alone would add roughly 15–18 minutes if representative. They may not represent
larger shards or sustained contention. Reserve **1–2 hours on CPU or 2–3 hours on GPU**
as a provisional planning budget, not a runtime guarantee. No cloud-dollar estimate
is implied for this local run.

The bounded decoded-patient cache is 256 MiB/four patients. Sampling plans, temporary
embeddings and framework overhead also consume host RAM; allow at least 2 GiB host
RAM for CPU execution and preferably 4 GiB for comfortable headroom. A patient shard
larger than the cache is processed transiently rather than retained. The entire
longitudinal window population is never materialized.

## Commands

From the repository root, the following **proposed future command was not executed**:

```powershell
& C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe -m src.architecture_study.cli run --config configs/model_studies/r0_r1_v1.toml --device cpu --output runs/model_studies/r0-r1-bounded-3fold-v1
```

Use `--device cuda` with a different new output directory for the GPU alternative;
do not run both merely to select a more favorable result. Existing output directories
are rejected. Failed runs remain incomplete for audit; this version does not resume
partial fits. The command trains only the six inner-training fits and produces
development estimates. It does not evaluate the canonical validation/test stream.

Reproduce the complete suite in the current machine's environment:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.tmp/study_dependencies')
$env:AI_CVD_TEST_EXTENSION_DIRECTORY = (Join-Path (Get-Location) '.tmp/duckdb_extensions')
& C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe -m unittest discover -s tests -v
```

For another machine, provision [requirements-study.txt](../requirements-study.txt)
and a matching DuckDB SQLite extension; omit the local `PYTHONPATH` override.
Future checkpoint consumers should call `artifacts.load_checkpoint` with the saved
study config and fold scaler rather than loading a bare state dictionary.

No known failing implementation gate remains. Remaining scientific limitations
include few independent training events, acquisition-process shortcuts, uncertain
generalization, retrospective support restrictions, and unmeasured full-study I/O
and convergence. The foundation is ready to run; neither model is yet validated
for clinical interpretation or deployment.
