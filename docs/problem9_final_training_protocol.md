# Problem 9: frozen final R0 training and validation protocol

Version 1.0.0, frozen before final fitting or canonical validation inspection.
Authoritative configuration: `configs/final_studies/r0_final_v1.toml`.
It binds the approved Problem 8 configuration by its semantic SHA-256, applies
explicit final-training overrides, and references the immutable canonical task
and dataset. Neither the original architecture study nor canonical data code is
modified. R0 is closed as the reference; there is no architecture search here.

## Populations and separation

Canonical dataset: `runs/level3-4h-attested-20260924T095744Z`.
Task: `level3_4h@2.1.1:4d8e27385d1187d649ee19ee63ec199dc40ee4fa6e05a5991ccbbb9bd73b6f13`.
Run metadata: `ba77a78193bdabea0b80e66ba7a3fdb663f7449a7109ead2357bd436aa02adec`.
TRAIN export: `b188c21388996630f26462b0c8e14462ee93dd3bf2462798168979c13d4317c4`.

The task loader supplies grid, lookback and horizon; model code does not copy
these constants. The referenced task is the five-minute, 96-step/eight-hour
history with a four-hour retrospective Level-3 alarm-initiation endpoint.
Source attestations, retrospective activity support restriction, censoring and
unknown clinical-history availability remain exactly as documented in Problem 7.

Three distinct roles must not be conflated:

1. Problem 8 training-fold comparisons are development architecture evidence.
2. Canonical validation is used only after final fitting for the prespecified
   intercept/threshold procedures below. Those results are development estimates.
3. Canonical test remains unopened. No test CLI exists in this package. Test
   evaluation is a separate later protocol and is not part of this execution.

Within canonical TRAIN, make a deterministic ten-way patient-grouped partition
using the approved event-presence-stratified hash assignment, seed 1729. Partition
0 (~10%) is reserved for label-blind SSL checkpoint scoring. The remaining ~90%
are the only SSL, scaler and head-fitting patients. Stratification uses training
event-presence flags only; sampled SSL windows are drawn without labels.
The reserved patients never enter gradients or head fitting. This deliberately
retains a patient-independent training checkpoint population at the cost of some
label exposure. There is no subsequent all-training refit, which would create an
unvalidated extra training phase. Exact memberships/counts are saved at execution.

## Architecture, inputs and preprocessing

Only the approved R0 is instantiated: primitive6_process25_v1 projection, a
32-unit GRU-D-style value encoder, evidence-masked temporal pooling, fixed process
means and two evidence flags; the SSL decoder has width 32 and is discarded for
risk inference. The risk head is one frozen-encoder L2 logistic layer.
Unused R1 process-width metadata remains inherited for compatibility, but no R1
branch or additional family is instantiated or fitted.

Six primitive channels: temperature, heart rate, SBP, DBP, SpO2 and valid Steps
increments. Derived clinical magnitudes, cumulative Steps, rolling summaries,
future labels and undated clinical history remain excluded. Exact primitive,
process and canonical feature orders are saved and checked.

Fit channel means/scales on unique bucket rows covered by 4,096 label-blind sampled
windows from the fitting patients, not on duplicated overlapping rows or validation.
This is the approved bounded scaler procedure, not the whole-run normalization
artifact. Save sample IDs, observation counts, unique-row population digest and
scaler hash. No threshold/calibration phase updates the scaler.

Natural validity M, artificial mask A and visible validity V=M(1-A) remain separate.
Hide 20% of naturally observed cells, retaining a visible value/target when possible.
Only A-and-M cells enter channel-balanced standardized Huber reconstruction loss
(delta 1). Naturally missing cells are never targets. Artificially hidden values
cannot enter the cache or reset visible recency. Empty-target batches skip the
optimizer step. Wholly unobserved primitive windows retain a deterministic zero
value pool, process context and absence flag; they remain in risk evaluation.

## Fixed SSL budget and checkpoint selection

Use **4,000 attempted SSL draws**, batch cap 64, four CPU threads, Adam learning
rate .001, gradient clipping norm 1, seed 1729. This is one prespecified doubling
of the bounded screen, not a budget search. Record actual contributing optimizer
updates separately: zero-target draws count toward the exposure ceiling but are
not falsely reported as gradient updates. Do not automatically extend this budget.

Training-only justification: all three screen R0 checkpoints selected draw 1,800.
From 1,000 to 1,800, checkpoint loss moved .19958→.19497, .21759→.20187 and
.24235→.23810. At 2,000 it was .19781/.20197/.23954. These curves show modest late
learning and noise, not proven convergence or a reason for unlimited SSL. The
extra fixed budget permits further representation learning while retaining a
checkpoint if later SSL deteriorates. It does not assert downstream benefit.

Checkpoint every 200 draws on **16 fixed label-blind batches** from the reserved
training patients, using identical artificial masks at every checkpoint. Select
the minimum mean nonempty-batch masked Huber loss, earliest exact tie. Initial
untrained loss is diagnostic, not a candidate checkpoint. This is an SSL stopping
rule within a frozen architecture, not a claim that reconstruction selects the
best Level-3 architecture. No head or validation performance selects the encoder.

Sampling retains patient-first UTC blocks, eight-hour block width from the approved
sampling definition, four patient visits per batch, at most 256 pooled windows
per patient, without replacement per cycle. Short pools are used without artificial
duplication. Save every sampled ID, draw probability, block, seeds, corruption
digest, target counts and complete checkpoint history.

Seeds: model/scaler 1729, SSL sampling 1829, checkpoint sampling 1929, training
corruption 2029, fixed checkpoint corruption 2129. Deterministic CPU algorithms
are enabled; package versions and thread count are recorded.

## Frozen risk head

Use fitting-patient labels only: four sampled positive windows per linked episode
set and one negative per patient, at most 20,000 samples, as in the approved
known-probability sampler. Fit inverse-inclusion-weighted logistic loss, normalized
to sum-one weights, with L2=1e-6 on standardized embedding coefficients. Estimate
embedding mean/scale on fitting examples with those same weights.

Use the numerically verified float64 LBFGS procedure: max 500 iterations,
tolerance_grad=1e-10, tolerance_change=1e-15, strong-Wolfe line search. Maximum
absolute standardized-coordinate gradient must be <=1e-8 or the run cannot
complete. Export coefficients into the frozen float32 risk model; save the double
coefficients/objective/gradient diagnostics separately. Preserve exact head input
IDs, labels, inclusion weights, embeddings and exported scores. Compare encoder
state before/after fitting and require checkpoint reload to reproduce scores.
No original study checkpoint or head is overwritten.

## Canonical validation, calibration and operating point

After the final model completion marker verifies, `score-validation` scores every
eligible canonical validation prediction with that model/scaler. It verifies the
canonical validation export, each shard hash, sample ID and outcome interval;
it has no test selection argument. No labels enter the forward pass. Every output
retains sample ID, patient manifest, prediction time, evidence flag and raw score.

Split validation patients into two disjoint groups by parity of the hash of
`[91027, patient_id]`. Even hashes calibrate; odd hashes develop the alert policy.
Both use complete eligible natural-prevalence streams, not balanced samples.
This avoids fitting calibration and selecting an operating point on the same
patients. It still does not make the policy-selected validation estimates unbiased.

Calibration is **intercept-only logistic recalibration**: add one intercept to
the clipped raw-score logit, with slope fixed to one. Fit unweighted natural-stream
Bernoulli likelihood via a deterministic bracketed root on [-40,40]. No class
balancing, architecture tuning or test labels. If a calibration partition lacks
either class, use identity calibration and explicitly record insufficient classes;
do not invent an estimated prevalence. Hitting the intercept bracket is a failure
requiring review, not silent saturation. Empty partitions fail.

On the separate policy partition, evaluate a fixed label-blind calibrated score
quantile grid: ten linearly spaced quantiles from 0 to .9; 101 log-tail quantiles
`1 - geomspace(.1, 1e-6, 101)`; and 1. Deduplicate and add a no-alert threshold
above one. Tail resolution is prespecified because a coarse .99-to-1 gap would be
inadequate for very rare outcomes and low alert budgets. Choose maximum
unique-episode sensitivity subject to **<=0.1 unmatched
alerts per supported patient-day**; ties prefer precision then the higher threshold.
This alert-burden cap is a frozen research operating target, not clinic-confirmed
capacity. Cooldown is fixed at four hours, not tuned. If no eligible policy events
exist, fail rather than invent a threshold. A no-alert selection is allowed and
must be reported honestly. No manual threshold adjustment after seeing results.

## Operational alert policy and metrics

Score every eligible five-minute prediction. An alert is emitted on a transition
from below threshold to at-or-above threshold, provided at least 240 minutes have
elapsed since the previous emitted alert. Equality at cooldown expiry is allowed.
A suppressed crossing still updates the above/below state: a continuously high
score does not emit again merely because cooldown expires. The first supported
score and the first score after a support gap are treated as new crossings if
above threshold; elapsed cooldown persists across gaps.

For each emitted alert at t, retrospectively match the earliest unmatched Level-3
episode with alarm initiation in **(t, t+horizon]**. An event at t is excluded;
one exactly at t+horizon is included. Each alert and episode match at most once.
An additional alert for an already matched event is unmatched, not another true
positive. Matching never changes alert emission/cooldown. No point adjustment.

Report unique-episode sensitivity among episodes with at least one eligible
forecast opportunity, alert precision, unmatched alerts per supported patient-day,
and lead-time quantiles to alarm initiation. Supported days are the union of
eligible prediction cells (count × task grid / 1,440), not enrollment duration or
calendar days with any measurement. Gaps are not negatives. Report 100% scoring
coverage only after matching every eligible indexed row, alongside supported-cell
fraction of each patient's first-to-last prediction span; the latter includes gaps
and is not a claim about true enrollment or complete clinical observation.

Evidence strata attribute alerts to physiology availability in the emitting
window. Episode denominators count episodes having an eligible prediction in that
stratum; an episode may have opportunities in both strata, so stratum denominators
are not additive. Only the stratum of the actual detecting alert receives a hit.
No-physiology windows may still contain Steps. Overall matching is unchanged;
strata do not rerun or reset the alarm policy. Retain process dependence as a
limitation even if global discrimination looks useful.

Validation artifacts contain calibration parameters, patient partitions, all
candidate thresholds/metrics, selected threshold, policy parameters, matched
alert/episode audit records and input/model hashes. These freeze the eventual
test operating point. No validation result authorizes further model search.
Raw and recalibrated window AP, AUROC, log loss and Brier diagnostics are also
reported on the policy partition, overall and by physiological evidence. Their
fixed .5 threshold diagnostics are distinct from the selected operational policy.

## Artifact integrity and interruption behavior

Every final fit records Git commit; complete effective config; task, canonical
run/TRAIN export and source-code hashes; feature order; scaler provenance; exact
patient/sample/exposure identities; seeds; Python/package versions; selected
checkpoint and head artifacts. Immutable completion manifests hash all generated
files except live `.log` output. Validate model/config/scaler/source identity on
reload. Live console logs must remain outside the output directory when possible.

The training `--resume` option **only verifies and skips an already completed fit**.
It does not restart or overwrite partial work. A partial directory fails loudly
and must be preserved for explicit recovery; automatic mid-fit resume is not
implemented. Validation stages likewise require new output directories and do
not support partial resume. This limitation is explicit, not a promise of
interruption-safe continuation. Completed artifacts are reusable and immutable.

## Manual execution

No private final fit or validation pass has been executed by the agent. Only
training-screen history files and existing provenance metadata were inspected.
Run the next training operation manually from the repository:

```powershell
Set-Location 'C:\Users\eldar\Projects\AI-CVD'
$finalLog = 'runs/final-r0-v1-training.log'
& 'C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe' -u -m src.final_study.cli train --config configs/final_studies/r0_final_v1.toml --output runs/final_studies/r0-final-v1 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    Add-Content -LiteralPath $finalLog -Value $line -Encoding utf8
}
if ($LASTEXITCODE -ne 0) { throw 'Final training failed; preserve output and log.' }
```

Estimated **45–90 minutes on four CPU threads**, allow two hours for cold private
shard I/O. This is a planning estimate, not a measured final-run time. Training
reports every 200 draws; preparation and head export can be quiet. Completion:
`runs/final_studies/r0-final-v1/complete.json`, successful exit, and
`Final R0 training complete: complete.json verified.` Preserve the entire run,
including `training_history.jsonl`, plus the external console log.

```bash
tail -F /c/Users/eldar/Projects/AI-CVD/runs/final-r0-v1-training.log
```

Verify/skip an already completed run with the same training command plus `--resume`.
If partial, do not delete/restart it blindly. Ask for recovery. A short verification
command is also available:

```powershell
& 'C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe' -m src.final_study.cli verify --output runs/final_studies/r0-final-v1
```

Only **after reviewing the completed final fit**, the following are the frozen
validation commands; neither has been run. Redirect/preserve console output using
the same PowerShell pattern above with distinct logs.

```powershell
& 'C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe' -u -m src.final_study.cli score-validation --model-run runs/final_studies/r0-final-v1 --output runs/final_studies/r0-final-v1-validation-scores
& 'C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe' -u -m src.final_study.cli develop-validation --model-run runs/final_studies/r0-final-v1 --predictions runs/final_studies/r0-final-v1-validation-scores --output runs/final_studies/r0-final-v1-validation-policy
```

Scoring is a full validation pass: conservatively budget **2–12 hours**, scaling
with eligible windows and CPU throughput; no validation counts were inspected to
refine that estimate. Policy development may take **10–60 minutes** and several
GB RAM because it evaluates fixed candidates on the full saved stream. Monitor
the saved console logs with `tail -F`; scoring reports each completed patient.
Each stage completes only when its own `complete.json` exists and hashes verify.
No partial resume; preserve all per-patient outputs, manifests, policies and logs.
There is deliberately no Problem 10/test command.

## Readiness

Infrastructure includes final training, verified inference, natural-stream
validation scoring, intercept calibration, threshold selection and episode-level
policy metrics. Synthetic tests exercise the complete training→reload→validation
scoring→calibration/policy path, alongside timing, isolation and tampering gates.
The approved R0 model/masking tests remain applicable unchanged.

Checks executed: all **37 model-study tests passed**, including eight new final
protocol tests. The final-specific suite was rerun after the last validation
diagnostic/provenance changes. No private dataset pass was used for these tests.

No scientific parameter choice remains open in this version. Final training,
convergence verification and validation development remain **unexecuted**, and
their outcomes must not be reported as successful in advance. Sparse events,
source-attested observability, retrospective support selection and observation
process dependence remain scientific limitations. Partial-job resume is an
engineering limitation. Problem 9 infrastructure is ready for the manual fit;
Problem 10 has not begun.
