# Canonical validation results — Problem 9 closed

**Final decision:** the manually completed full audit passes verification.
Problem 9 is fully closed. The scientific evaluation specification is immutable;
validation performance does not justify changing the model, calibration or operating
point before test. A separately bound guarded CPU acceleration adapter is approved.
No canonical test data were accessed or evaluated, and no model was retrained.

The final results below supersede the historical pending-audit notes retained later
in this document. Do not rerun the full audit against its existing directory.

## Verification and immutable bindings

All eight final audit artifact checksums pass, including the five saved audit work
arrays, result report, protocol and evaluation specification. The audited source
hashes still match the code that ran. The passing audit verified **all 40,199,264
validation prediction IDs**, their ordering and source-derived patient/time identity,
labels and horizon linkage, source/prediction file hashes, and physiological masks.
Every operational threshold candidate was replayed; the selected per-patient alert
records exactly matched the frozen implementation.

The canonical validation population consists of **1,871 patients**, separate from
all final fitting/checkpoint patients. Calibration uses the predetermined **937**
patients; threshold development uses the other **934**, with no overlap. Calibration's
likelihood equation and sample counts passed independent checking. The operational
threshold follows the frozen constraint, candidate grid and tie-breaks.

Current model/head/scaler hashes, task/run/spec fingerprints, calibration parameters,
threshold, cooldown, matching and software provenance match the final specification.
The completed audit ran validation-only code; neither its entry points nor this
review loaded a canonical test export or test-patient outcome. This verifies this
workflow, not unrelated activity elsewhere on the machine.

Immutable scientific specification:
`runs/final_studies/r0-final-v1-validation-audit/final_evaluation_spec.json`

SHA-256: `0ad2c76b2f8ec65ead77c66067e3760d3581e6d762b47a1ed2e3cb5a6986899c`.
The private audit completed in **416.27 seconds**. This review checked its completion
artifacts and provenance rather than repeating another full private-data pass.

## Natural-stream discrimination

| Population | Patients | Windows | Positive windows | AUROC | AP | Prevalence | AP/prevalence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entire validation | 1,871 | 40,199,264 | 1,068 | .605739 | 4.94050e-5 | 2.65677e-5 | 1.8596 |
| Calibration partition | 937 | 19,883,477 | 480 | .670676 | 8.05373e-5 | 2.41406e-5 | 3.3362 |
| Policy-development partition | 934 | 20,315,787 | 588 | .552291 | 3.65951e-5 | 2.89430e-5 | 1.2644 |

These are all eligible windows with natural prevalence, not the weighted sampled
training-fold estimates from Problem 8. There are **23 unique eligible Level-3
episodes in 20 event patients**; 1,068 positive windows are repeated prediction
opportunities, not 1,068 clinical events.

Compared descriptively with development AUROC ~.631 and AP ~4.68e-5, validation
AUROC is ~.025 lower while AP is slightly higher. AP depends on prevalence and
case mix; the difference cannot be attributed to the 4,000-draw budget. Different
patients, fitting populations and evaluation sampling prevent a controlled budget
comparison. No significance test or improvement claim is made.

**Interpretation:** modest ranking signal generalized, but less convincingly than
the development screen suggested. The .671 versus .552 partition variation is
substantial, with only ten event patients in each partition. AP is descriptively
1.86 times prevalence, above the prevalence baseline, but its absolute value is
only .00494%. This is not evidence of strong clinical discrimination.

## Calibration

The frozen intercept-only fit is **-0.26813403062419017** (odds multiplier ~.765).
It was estimated solely on calibration-partition patients. Ranking is unchanged:
raw and calibrated AUROC/AP are exactly equal in all audited strata.

| Population | Observed prevalence | Raw mean risk | Calibrated mean risk | Raw log loss | Calibrated log loss | Raw Brier | Calibrated Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entire validation | 2.65677e-5 | 3.14827e-5 | 2.40784e-5 | .000305351 | .000305070 | 2.65669085e-5 | 2.65668178e-5 |
| Calibration partition | 2.41406e-5 | 3.15641e-5 | 2.41406e-5 | .000276694 | .000275743 | 2.41397520e-5 | 2.41396952e-5 |
| Policy partition | 2.89430e-5 | 3.14030e-5 | 2.40174e-5 | .000333398 | .000333773 | 2.89424162e-5 | 2.89422925e-5 |

Raw mean risk overestimates whole-validation prevalence by ~18.5%; calibrated mean
risk underestimates it by ~9.4%. The intercept matches prevalence in its fitting
partition by construction, but underestimates policy-partition prevalence by ~17%.
Policy-partition log loss becomes slightly worse. This is imperfect transport of
calibration across patients, not a ranking improvement. Tiny Brier values reflect
the extremely rare outcome and do not establish good calibration. Reliability
throughout the high-risk tail remains uncertain with only 23 events; no additional
calibration fitting or tuning is authorized.

## Frozen alert-policy performance

Calibrated threshold: **0.00007816438698653833** (~.0078164% window risk).
The unchanged rule emits the first threshold crossing, enforces a four-hour
cooldown, and matches each emitted alert to the earliest unmatched Level-3 episode
in `(t,t+4h]`. No point adjustment or window-recall substitution was used.

| Population | Supported patient-days | Eligible episodes | Detected | Sensitivity | Alerts | Matched | Unmatched | Unmatched/day | Alert precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Policy-development partition | 70,540.927 | 13 | 1 | 7.6923% | 6,952 | 1 | 6,951 | .0985385 | .0143843% |
| Calibration partition, descriptive | 69,039.851 | 10 | 0 | 0% | 7,606 | 0 | 7,606 | .1101683 | 0% |
| Entire validation, descriptive | 139,580.778 | 23 | 1 | 4.3478% | 14,558 | 1 | 14,557 | .1042909 | .0068691% |

The <=.1/day constraint **was met in the designated threshold-development
partition**. It was not met across the entire validation population (.10429/day)
or the separate calibration partition (.11017/day). This is a generalization
limitation, not a threshold-selection implementation defect. Do not retune it.

There was **one detected episode**. Its lead time to alarm initiation was
**51.9167 minutes (51 min 55 s)**. Median, Q1, Q3, minimum and maximum are consequently
all 51.9167 minutes; the zero-width IQR is not evidence of consistent warning time.
This interval is potentially actionable in isolation, but one detection does not
establish reliable warning. It is not lead time to ambulance dispatch, staff
escalation decision or physiological crisis onset.

Scoring coverage is **100% of canonical supported eligible cells**. The audited
within-patient first-to-last prediction span coverage is also 1.0. This does not
mean all enrolled time was observed: run-in, censoring and the retrospective activity
support restriction define the denominator. The reported 139,580.778 patient-days
are summed eligible grid time, not confirmed enrollment follow-up.

Operational usefulness is **not demonstrated**: one true alert among 14,558 alerts
and detection of 1/23 episodes is a poor tradeoff for this endpoint, even though
the selection partition satisfied its alarm-burden constraint. Unmatched alerts
mean unmatched to the defined Level-3 endpoint, not proof that every such alert
lacked any clinical significance.

## Evidence availability

Whole-validation descriptive strata:

| Stratum | Windows | Positive windows | Supported days | AUROC | AP | AP/prevalence |
|---|---:|---:|---:|---:|---:|---:|
| Physiology observed | 30,715,660 (76.41%) | 936 | 106,651.597 | .548658 | 5.14483e-5 | 1.6883 |
| No physiology | 9,483,604 (23.59%) | 132 | 32,929.181 | .817030 | 7.34032e-5 | 5.2737 |
| No primitive observations | 9,115,474 (22.68%) | 115 | 31,650.951 | .812602 | 5.85858e-4 | 46.4381 |

The no-primitive stratum is nested within no physiology; it is not an additional
disjoint population. It includes no observed Steps increment either. Its 115
positive windows are not 115 independent episodes. Unique episode counts for this
additional descriptive stratum were not persisted, so they are not inferred from
window counts.

In the policy-development partition, physiology-observed alert windows generated
6,926 alerts, one matched, .129016 unmatched alerts per stratum-supported day;
sensitivity was 1/13 and precision .0144383%. No-physiology alert windows generated
26 alerts, none matched, .00154160 unmatched/day, with 0/3 eligible episodes
detected in that stratum. Episode opportunity sets can overlap across evidence
strata, so their denominators are not additive. Operational no-primitive metrics
were not part of the saved predefined alert-stratum report and remain unavailable.

Evidence availability clearly changes observed performance, but the small event
counts make attribution uncertain. Stronger ranking when no primitive values are
observed is particularly important: R0's value pool is then exactly zero, and its
remaining predictive information is observation/recency/context. This supports
retaining observation-process dependence as a major scientific limitation and
baseline. It does not establish a causal explanation or a robust subgroup benefit.

## Scientific conclusion and closure

1. **Generalization:** modest above-baseline ranking persisted; overall AUROC .606
   weakens the ~.631 development evidence, with pronounced partition variability.
2. **4,000-draw benefit:** no material ranking improvement is established. This is
   not a matched 2,000-versus-4,000-draw comparison.
3. **AP:** descriptively above prevalence (1.86x), but absolute AP and operational
   precision remain extremely low; significance is not claimed.
4. **Clinical usefulness:** not established at the frozen burden target. Only
   one of 23 validation episodes was detected.
5. **Warning:** ~52 minutes for that single detection; no population-level warning
   distribution can be established from one event.
6. **Strongest result:** an auditable, leakage-controlled natural-stream benchmark
   reveals limited operational utility and substantial observation-process signal
   despite apparently encouraging development discrimination. The methodological
   rigor is stronger than a claim of a new clinically effective detector.
7. **Main weakness:** very few independent events, weak operational precision and
   sensitivity, and uncertain transportability of observation-driven signals.

**Problem 9 is closed.** Preserve R0, its head/scaler, intercept, threshold and
policy exactly. Disappointing validation is not a reason to reopen model selection.
No future test result may be used to alter them. The eventual held-out test is an
evaluation of this frozen system, including a potentially negative clinical result.

## Bound engineering supplement: guarded CPU batch 256

The original 768-window benchmark was expanded, without outcome-based selection,
to the **same first three full validation patient trajectories: 76,413 windows**.
Source/prediction IDs and ordering, feature projection, checkpoint/head and
calibration were unchanged. All candidate predictions were compared with audited
saved reference scores. Repeated accelerated inference was deterministic.

- Predeclared tolerance: `abs_error <= 1e-10 + 1e-5 * abs(reference_score)`.
- Maximum absolute error: **1.12777e-10**, within that combined tolerance.
- All threshold classifications and complete alert-policy records matched exactly.
- The subset contained one emitted alert, unmatched; it contains no detected event,
  which limits empirical matched-event coverage. Separate synthetic policy tests
  cover episode matching and boundaries.
- Guarded throughput: **3,121–3,448 windows/s**; ~23.0 s for one full subset pass.
  Entire two-pass equivalence check: **45.73 s**. No full validation rerun occurred.

The adapter fixes CPU/four threads/batch 256, with original batch-64 replay for each
patient's first/last reference batch and all reference batches near the inverse-
calibrated threshold. Replayed scores must pass tolerance, be deterministic, and
replace candidate scores exactly. Any failed canary aborts evaluation; tolerances
must not be relaxed after inspecting test outcomes. This is finite-subset engineering
equivalence, not a proof of bitwise identity for unseen samples.

**Approved:** `cpu256_guarded_v1`, bound in
`runs/final_studies/r0-final-evaluation-engineering-v1.json` to the original immutable
scientific specification and model hashes. The supplement changes only inference
execution, not scientific definitions. It records its exact source hashes and Git
base; the adapter is committed alongside this report. GPU and unguarded alternative
thread/batch settings are not approved. The original scientific JSON is unmodified.
Supplement SHA-256:
`d284a32cbf27a315f72cd8a8da8dd7bd4cd622cadf3ccae19d831221e3e5be83`.

Checks this turn: completed-audit manifest and binding verification passed; the
76,413-window validation-only equivalence check passed; two new guard/rejection
unit tests and four existing audit tests passed. No private optimization, head
fitting, calibration fitting or policy selection was rerun.

## Problem 10 handoff — not executed

The exact next step is to implement/review a one-shot held-out test runner that
consumes the immutable scientific specification **and** its engineering supplement,
enforces hashes and guard failures, scores every eligible test window in canonical
order, and applies the already frozen calibration and operational policy. It must
report natural-stream discrimination/calibration, unique-episode sensitivity,
precision, unmatched alerts/day, coverage, warning times and fixed evidence strata,
with patient/event-aware uncertainty clearly distinguished from independent windows.
It must not fit, select, recalibrate or tune anything on test.

No test-run command is issued because that bound runner has not yet been implemented
or reviewed. Implementing it is the next task; do not execute test automatically.
Provide its exact manual command, runtime estimate, resume behavior, logs and
completion markers only after that implementation passes synthetic tests.

## Historical pending-audit notes (superseded)

The scoring and policy stages have completion markers. **Problem 9 validation
is not yet closed, and the final test specification is not yet frozen.** The user
requires checks to pass before interpretation and prohibits automatic long or
full private-data passes. Complete prediction/source ID verification spans 1,871
patients, 40,199,264 eligible windows and a 2,381,946,142-byte prediction directory.
It has therefore been
prepared as a manual audit, not silently omitted or represented as complete.

No model was retrained, no frozen training/policy file changed, and no canonical
test export, patient shard or outcome was accessed. Final model verification still
passes. Added audit/profile utilities live outside the frozen source directories.

## Checks completed now

- Both success markers exist. The four policy artifact hashes pass.
- Score manifest/provenance hashes and their links to canonical validation export
  metadata pass. Each listed patient's sample count, episode list and source-shard
  hash agree with that export; all score-file hashes are represented in the marker.
- Validation patients satisfy the canonical validation assignment, are unique and
  disjoint from final fitting and checkpoint-holdout populations.
- Calibration and policy patients exactly match the frozen hash partitions and
  have no overlap. The recorded calibration and policy source hashes match the
  unchanged frozen implementation.
- Model completion, final specification, source implementation and canonical
  dataset/export bindings match. No model/head/scaler substitution was found.
- The recorded selected candidate obeys the frozen objective, tie-breaks and
  <=0.1 unmatched alerts/day constraint **within the saved candidate table**.
- Three fixed validation patients (first 256 eligible rows each) passed source and
  prediction file hashes, source prediction-ID/time alignment, and reference
  score reproduction. Selection was independent of outcomes.

These metadata/subset checks do **not** establish all-patient prediction checksum,
ID/label alignment or independent policy replay. Those remaining gates are required
before reporting scientific conclusions or freezing test evaluation.

The saved `policy_development_metrics.json` describes the **threshold-development
partition**, not the entire validation cohort. It must not be labeled whole-stream
validation AUROC/AP. Whole-validation calibration diagnostics also include patients
used to fit the intercept. Both distinctions will remain explicit in the report.

## Manual audit and conditional freeze

`scripts/audit_validation.py` performs no training or model scoring. It reads only
canonical validation source shards and saved validation scores, and will:

1. Verify every prediction and source-shard checksum, row ordering/count, canonical
   sample ID from patient/time, source label, horizon-derived label and evidence mask.
2. Verify the intercept's calibration-population likelihood equation without refitting.
3. Reconstruct the exact label-blind candidate quantiles, replay every candidate's
   first-crossing/cooldown/one-to-one matching results, and compare the selected
   per-patient alert/episode records with the frozen outputs.
4. Recompute natural-stream AP, tied-score AUROC, prevalence, AP/prevalence, raw and
   calibrated mean risk, log loss and Brier score. Report the full validation stream,
   each validation role, physiology/no-physiology and no-primitive strata separately.
5. Report whole-stream and role-specific episode counts, event patients, episode
   sensitivity, alert counts/precision/burden, eligible patient-time and lead times.
   Full-validation operating metrics are descriptive because the policy was chosen
   on a subset of those patients; policy-partition estimates remain development.
6. Only after all gates pass, create `final_evaluation_spec.json` binding the model,
   head, scaler, calibration, threshold, policy/metric definitions, task/run fingerprints,
   validation audit/provenance and software state. It does not access or execute test.

Run manually:

```powershell
Set-Location 'C:\Users\eldar\Projects\AI-CVD'
$auditLog = 'runs/final-r0-v1-validation-audit.log'
& 'C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe' -u -m scripts.audit_validation --output runs/final_studies/r0-final-v1-validation-audit 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    Add-Content -LiteralPath $auditLog -Value $line -Encoding utf8
}
if ($LASTEXITCODE -ne 0) { throw 'Validation audit failed; preserve outputs and log. Do not run test.' }
```

Estimated **30–120 minutes**, potentially longer for cold I/O and tens of millions
of ID hashes. Budget several GB RAM and roughly 12 bytes per validation window
plus audit reports for memory-mapped work arrays. This estimate is unmeasured;
the audit uses saved scores and does not repeat the 12–13-hour inference pass.
It logs each patient and each discrimination calculation.

**Resume:** no partial resume; existing output directories are rejected and never
overwritten. Preserve interrupted output for diagnosis. Do not delete original
model, scoring or policy artifacts. Preserve the complete audit directory and log,
including work arrays, which are provenance-linked and not automatically removed.

Success requires a zero exit, `validation_results.json`,
`final_evaluation_spec.json`, and `complete.json` under
`runs/final_studies/r0-final-v1-validation-audit`, plus the console message
`Validation audit complete; final_evaluation_spec.json frozen. No test evaluation executed.`

```bash
tail -F /c/Users/eldar/Projects/AI-CVD/runs/final-r0-v1-validation-audit.log
```

This is the next command, **not a canonical test command**. After it passes, review
the validated results and finalize the scientific interpretation requested by the
user. Do not adjust the model or operating point in response to disappointing results.

## Bounded scorer profile

Fixed subset: first three validation manifest patients, first 256 rows each;
768 windows total. The reference reproduces saved scores exactly. Tolerance was
fixed in the harness before execution: absolute 1e-10 plus relative 1e-5, with a
separate requirement for identical alert-policy records on the subset. All tested
paths were deterministic on repeated inference and met those subset gates.

| Path | Examples/s | Max absolute score error vs saved | Bitwise exact? |
|---|---:|---:|---|
| Original CPU, 4 threads, batch 64 | 1,248 | 0 | Yes |
| Vectorized windows, CPU, 4 threads, batch 64 | 1,189 | 0 | Yes |
| Vectorized windows, CPU, 1 thread, batch 64 | 1,902 | 4.00e-11 | No |
| Vectorized windows, CPU, 4 threads, batch 256 | 3,365 | 4.00e-11 | No |
| Vectorized windows, GPU, batch 256 | 961 | 8.37e-11 | No |

Reference preparation took .067 s and forward passes .548 s; loading/hashing the
three source/prediction pairs took .110 s. CPU inference dominates this subset;
vectorized extraction alone does not solve it. Increasing batch size looks more
promising than simply increasing threads. The GPU measurement includes cold-start
effects and many small recurrent operations, so it is not a warmed-up throughput
or full-run forecast. Peak allocated GPU memory was approximately 50.2 MiB.

At the original measured rate, tens of millions of windows can plausibly explain
an hours-long pass. The benchmark does not cover all original per-window ID/label
checks or output serialization, nor cold-cache whole-cohort I/O, so it cannot
attribute the entire 12–13-hour duration to one component.

**No accelerated scorer is approved for canonical test yet.** CPU batch 256 is a
candidate with approximately 2.7x subset throughput, not a guarantee of identical
threshold decisions across an unseen full stream. Small score errors could matter
near the frozen threshold. The conditional evaluation freeze retains the reference
CPU batch-64/four-thread path. Do not silently substitute GPU/thread/batch settings.

Private profile artifact: `runs/final_studies/r0-validation-subset-profile.json`.
The original and optimized calculations were compared on this same fixed subset;
no additional full validation pass was run for benchmarking.

## Tests and remaining work

Four quick tests pass: vectorized input exactness/future-prefix isolation; randomized
alert-policy equivalence including ties, gaps and boundaries; independent natural
ranking/probability metrics; and a fictional end-to-end audit and conditional freeze.
The tests uncovered and corrected unsigned-label negation in the **new independent
audit calculator** before any private full audit; frozen validation metrics/code
were not changed.

AUROC, AP, prevalence ratio, calibrated-risk conclusions, final threshold/episode
sensitivity, unmatched-alert rate, precision and lead-time interpretation remain
**pending verified full-audit results**. No significance claim or development-to-
validation comparison is made prematurely. A 4,000-draw versus 2,000-draw causal
improvement claim would in any case require a matched comparison, not merely two
different patient populations.

Problem 9 validation is not fully closed. The final evaluation specification is
implemented as an audit-gated immutable output but has not yet been generated on
private validation. No architecture search, protocol change or Problem 10 execution
has occurred.
