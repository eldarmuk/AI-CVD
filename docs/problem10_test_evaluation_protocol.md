# Problem 10: one-shot held-out evaluation protocol

The evaluation runner is implemented; canonical test scoring has **not** been
executed. Problem 9 remains closed. No model, calibration, threshold, cooldown,
projection, task or architecture changes are permitted based on test results.

## Frozen inputs and engineering approval

- Model: `runs/final_studies/r0-final-v1`.
- Dataset: `runs/level3-4h-attested-20260924T095744Z`.
- Scientific specification:
  `runs/final_studies/r0-final-v1-validation-audit/final_evaluation_spec.json`,
  SHA256 `0ad2c76b2f8ec65ead77c66067e3760d3581e6d762b47a1ed2e3cb5a6986899c`.
- Engineering supplement:
  `runs/final_studies/r0-final-evaluation-engineering-v1.json`,
  SHA256 `d284a32cbf27a315f72cd8a8da8dd7bd4cd622cadf3ccae19d831221e3e5be83`.

The original scientific specification is unchanged. Its original batch-64
scorer description is supplemented only by the separately bound batch-256
approval. CPU, four threads, deterministic inference, unchanged R0 and head,
and the approved tolerance `atol=1e-10, rtol=1e-5` are mandatory. The approved
adapter repeats batch-64 canaries and batches near the inverse-calibrated
threshold. Discrepancies abort; neither tolerance nor policy may be relaxed.
This is bounded engineering equivalence, not proof of bitwise identity on every
unseen window. GPU and other thread/batch settings are not approved.

## Evaluation-only boundary and one-shot safeguards

`scripts/test_evaluation.py` exposes only `preflight` and `run`. It has no fitting,
optimizer, gradient, calibration-estimation, threshold-search, policy-search or
architecture-selection branch. Model construction loads the original frozen
checkpoint format; it does not call training. All model parameters have gradients
disabled, and execution is enclosed in `torch.inference_mode()`.

The output location is fixed to `runs/final_studies/r0-final-v1-test`. It must not
exist, even partially. There is no output override and no resume flag. A clean
Git working tree and an exact preflight receipt for that commit and source are
required. The output directory is exclusively reserved before the first outcome
is decoded. `started.json` records that transition. An interruption preserves
all partial files; absence of `complete.json` means incomplete, not permission
to restart. Do not rename/delete the directory to bypass the gate. Recovery
requires explicit review of what was accessed and saved.

Preflight checks the pinned scientific and engineering documents; original
model manifest/checkpoint/head/scaler hashes; checkpoint configuration; task;
feature order; frozen source hashes; approved Python/package versions;
validation provenance hashes; canonical run and test export hashes; unique
test membership and disjointness from training/checkpoint and validation
patients. The audited metric helper is also pinned by hash. New runner source
hashes and Git commit are bound in the receipt and final result manifest.

The calibration intercept, threshold, four-hour cooldown and matching rule are
loaded from the frozen specification. No CLI override exists. Scientific
specification fields describing the earlier validation-selection procedure are
retained as provenance only; the runner never invokes that procedure.

## Outcome-blind preflight

Preflight passed during implementation: **1,839 test patients and 39,284,268
eligible prediction windows** are planned. These are population metadata,
not test performance results.

The canonical JSON export mixes non-outcome metadata with event data. Preflight
hashes/scans its bytes but lexically skips unrequested values without decoding
them. Only patient IDs, split, counts, paths, hashes and schema are parsed.
No patient shard is opened; no test label, event, prediction, prevalence or
performance statistic is inspected. The same selective mechanism reads only
artifact hashes from canonical run metadata.

Preflight cannot prove per-row sample identity, label correctness, event linkage,
actual shard contents, evidence flags, or numerical inference on unseen patients.
Those are verified during the authorized run: each shard hash is checked before
opening, every sample ID is recomputed from patient and prediction time, indexed
lookback bounds and ordering are checked, and labels/event-index ranges are
reconstructed using `(t,t+horizon]`. The canonical support/censoring construction
is inherited through the immutable dataset fingerprint, not redefined here.

## Metrics and event accounting

All eligible test windows are scored in immutable export order. Raw and calibrated
scores, sample IDs, timestamps, labels, patient membership, evidence masks and
event linkage indices are saved. Patient records retain individual episodes,
emitted alert timestamps, matches and lead times.

Discrimination uses the same exact tied-score AUROC/AP and probability diagnostics
as validation. Report natural prevalence, AP/prevalence, raw and calibrated mean
risk, log loss and Brier score, with the frozen intercept. Ranking confidence
intervals refer to raw-score ranking; separately report any calibrated ties.

Operational evaluation uses first threshold crossing, reset of crossing state
at support gaps, a four-hour cooldown that is not reset by gaps, and one-to-one
matching to the earliest unmatched episode in `(t,t+4h]`. Sustained above-threshold
risk does not emit a new alert merely because cooldown expired. No point
adjustment is performed. Episodes must have at least one eligible forecast
opportunity. Report unique episodes and event patients separately from positive
windows, detected episodes, sensitivity, precision, unmatched alerts per supported
patient-day and lead-time minimum/Q1/median/Q3/maximum. Lead time ends at alarm
initiation retrospectively classified Level 3, not dispatch or crisis onset.

Supported patient-days equal eligible grid cells times the canonical grid duration
divided by one day; unsupported gaps are excluded. Complete scoring gives 100%
coverage of eligible support, not verified enrollment or continuous outcome
surveillance. The within-patient-span coverage measure is reported separately.

Evidence strata are physiology-observed, no-physiology and nested no-primitive.
Discrimination uses all windows in each stratum. Operational subgroup analysis
attributes the **existing global-policy alerts** to evidence at emission; it never
reruns crossing/cooldown on a filtered stream. Subgroup episode denominators are
unique episodes with at least one eligible forecast opportunity in that stratum.
They can overlap and must not be added. Empty/undefined quantities are null,
not zero performance claims.

## Prespecified uncertainty

`scripts/test_uncertainty.py` consumes only sealed saved results. It does not
rescore, access canonical shards, or refit anything. A separate immutable
`uncertainty-v1` directory is reserved; rerun/resume is refused.

The fixed procedure is **200 ordinary patient-cluster bootstrap replicates**,
seed **101027**, with 95% percentile intervals. A sampled patient contributes
all its windows, episodes, alerts and supported time together. Ranking uses
exact integer-weighted tied-score AUROC/AP, with one global score sort. Episode
sensitivity and alert burden use the same patient multiplicities. Per-replicate
results, draw indices and seed are retained. Undefined replicates are counted
and excluded from quantiles, with their count disclosed.

A supplementary Clopper–Pearson episode-sensitivity interval is reported; it
assumes independent episodes and may understate dependence when patients have
repeated events. Patient-cluster intervals are primary. Sparse event patients,
heterogeneity and only 200 replicates limit tail precision. Neither window-level
IID intervals nor post-hoc bootstrap expansion are permitted by this version.

## Validation comparison and interpretation

The result JSON embeds the frozen validation comparator. Compare discrimination,
calibration, observation-process dependence, episode sensitivity, alert burden
and lead times descriptively. Validation AUROC was 0.60574, AP 0.000049405,
prevalence 0.000026568; one of 23 episodes was detected among 14,558 alerts,
with 0.10429 unmatched alerts per supported patient-day over all validation.

Do not call differences significant without an appropriate analysis. Numerical
improvement alone does not establish clinical utility. No comparison permits
retuning. Remaining source-attestation, retrospective-support selection and
observation-process limitations persist even if test discrimination improves.

## Tests and audit evidence

Nine new synthetic tests cover selective outcome skipping, no fitting/gradient
calls, failed-preflight access gating, immutable output and tamper rejection,
full fictional scoring/reload/sealing, separate uncertainty sealing, sample
alignment, frozen calibration, cooldown and open/closed horizon boundaries,
single-use event matches, support denominators, subgroup attribution, zero Steps,
empty physiology, and bootstrap equivalence to explicitly duplicated patients.

Two existing acceleration tests verify deterministic CPU64/CPU256 tolerance,
near-threshold reference replay, bad-canary rejection and ordering. Four audit
tests cover exact policy equivalence with ties/gaps, causal vector windows and
fictional audit integration. Eight final-study tests cover frozen definitions,
policy, scope, artifact integrity and fictional fitting/reload (no private fit).
All are quick, synthetic tests. No test patient shard is used in these checks.

## Manual commands and completion criteria

Run from `C:\Users\eldar\Projects\AI-CVD` in the approved environment. The final
post-commit preflight receipt is saved at the following path. If repository state
changes after receipt creation, execution fails; review the change before making
a new immutable receipt. Preflight itself takes about 10 seconds:

```powershell
& C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe -u -m scripts.test_evaluation preflight --receipt runs/final_studies/r0-final-v1-test-preflight-v1.json
```

Once that receipt exists, the definitive command is:

```powershell
& C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe -u -m scripts.test_evaluation run --receipt runs/final_studies/r0-final-v1-test-preflight-v1.json --acknowledge-one-shot 2>&1 | Tee-Object -FilePath runs/final_studies/r0-final-v1-test.console.log
if ($LASTEXITCODE -ne 0) { throw 'Test evaluation failed: preserve all artifacts for recovery review; do not restart.' }
```

Allow approximately **4–7 hours**, including hashing, identity checks, output
serialization and exact metrics. The small validation benchmark suggests roughly
3.2–3.5 hours of inference alone; thermal throttling and I/O can increase this.
This is an estimate, not a test benchmark. Keep at least 8 GB RAM available and
5 GB free output storage; do not run the bootstrap concurrently with scoring.

Success requires `runs/final_studies/r0-final-v1-test/complete.json` with
`status: complete` and matching artifact hashes, plus the terminal message
`TEST EVALUATION COMPLETE`. `started.json`, individual patient files or the last
scoring progress line do not prove completion. The immutable result manifest is
`final_result_manifest.json`; aggregate results are `test_results.json`.

Monitor from another Bash window:

```bash
cd /c/Users/eldar/Projects/AI-CVD
tail -f runs/final_studies/r0-final-v1-test.console.log
```

After the scoring completion marker has been verified, run the separate manual
uncertainty operation (estimated **1–4 hours**, potentially longer under memory
pressure; no automatic execution):

```powershell
& C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe -u -m scripts.test_uncertainty --acknowledge-long-analysis 2>&1 | Tee-Object -FilePath runs/final_studies/r0-final-v1-test-uncertainty.console.log
if ($LASTEXITCODE -ne 0) { throw 'Uncertainty analysis failed: preserve partial output for review.' }
```

Its completion marker is `r0-final-v1-test/uncertainty-v1/complete.json` and
terminal message `TEST UNCERTAINTY COMPLETE`. Monitor with `tail -f` on the
uncertainty console log. There is no automatic resume for either command.

Preserve both console logs, the receipt, original frozen model/spec/supplement,
all patient prediction and event records, pooled arrays, manifests, completion
markers and bootstrap outputs. Console logs are outside the sealed output so a
still-open log cannot invalidate an artifact checksum. All private artifacts
remain Git-ignored. Commit only implementation, tests and this protocol.
