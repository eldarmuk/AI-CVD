# Problem 10 held-out results: verification status

**Status: execution completed; final independent row/checksum/metric audit and
patient-cluster uncertainty are pending. No test metrics have been interpreted
for this report.** No model inference, fitting, recalibration, threshold selection,
policy change or test rerun was performed during this review.

## Verified completion metadata

The quick read-only checks passed for
`runs/final_studies/r0-final-v1-test`:

- `complete.json` exists with `status: complete`.
- The recovery console contains `TEST EVALUATION COMPLETE`. The PowerShell log
  is UTF-16; encoding-aware reading is required.
- The saved population metadata identifies 1,839 patients and 39,284,268 windows,
  in the same patient order as the canonical test export.
- Checksums for the final manifest, started record, patient manifest and recovery
  record passed. Frozen checkpoint/head/scaler, model seal, scientific specification,
  engineering supplement, task/export fingerprints and scoring source bindings
  passed the metadata preflight.
- The final result manifest contains the original preflight receipt and exact
  frozen scientific/engineering specifications. The recovered 391-patient prefix
  hashes are retained in the final seal without changes.
- Recovery provenance preserves the accidental interruption, original receipt,
  audited reuse and the user's attestation of no test-driven scientific changes.
- The original evaluation and recovery code have no fitting/tuning path. The
  source hashes match the completed execution; CPU batch 256 and its approved
  guards remain fixed.
- `uncertainty-v1` does not yet exist.

These are completion/provenance checks, **not a claim that the independent full
sample-level audit has already passed**. Per-patient guard records and all saved
prediction/source bytes still need complete verification by the command below.

## Why the results table is intentionally pending

The user's standing constraint prohibits automatically running operations longer
than approximately five minutes or full private-data passes. Verifying all
39,284,268 sample identities, labels, pooled-array entries and event links is a
full private-data pass. Based on the previous 391-patient audit taking about
90–100 seconds, a complete audit plus exact metric arithmetic is expected to take
approximately **10–25 minutes**, with additional variation from disk/memory load.

The user also requires verification before interpretation. Consequently AUROC,
AP, prevalence, calibration, operational metrics, evidence-stratum results and
paper-ready conclusions are deliberately not copied or interpreted yet. Pending
is not zero, and no finding is inferred from the completed-run flag alone.

## Read-only full audit command

`scripts/audit_test_completion.py` independently checks every sealed artifact hash,
patient record, canonical source shard, exact sample ID/cutoff/order, label and
episode linkage, evidence mask, raw/calibrated score alignment, pooled cache,
scoring guard configuration, operational-policy replay and aggregate metric
arithmetic. It preserves all original test artifacts and writes a separate
immutable audit directory. It does not perform model inference or fit anything.
Two quick synthetic tests passed, including a complete fictional audit, checksum
tamper rejection, overwrite refusal, no-inference enforcement and console encodings.

Run manually from `C:\Users\eldar\Projects\AI-CVD`:

```powershell
& C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe -u -m scripts.audit_test_completion --output runs/final_studies/r0-final-v1-test-audit-v1 2>&1 | Tee-Object -FilePath runs/final_studies/r0-final-v1-test-audit-v1.console.log
if ($LASTEXITCODE -ne 0) { throw 'Read-only test audit failed. Preserve artifacts and logs; do not rerun scoring.' }
```

Completion requires `r0-final-v1-test-audit-v1/audit_results.json` with
`status: passed`, its sealed `complete.json`, and the terminal message
`TEST RESULT AUDIT COMPLETE`. Existing/partial audit output is never overwritten;
there is no automatic resume. A failed audit requires review and, if authorized,
a new audit directory, never a test-scoring restart.

Monitor from Bash:

```bash
cd /c/Users/eldar/Projects/AI-CVD
tail -f runs/final_studies/r0-final-v1-test-audit-v1.console.log
```

Preserve the entire original test directory, frozen inputs, preflight receipt,
original/recovery console logs and provenance, plus the new audit directory/log.

## Pending fixed uncertainty analysis

After the full audit passes, run the existing frozen 200-replicate patient-cluster
bootstrap, seed 101027. This analyzes sealed predictions only, performs no model
scoring or refitting, and cannot trigger scientific changes. Expected runtime is
**1–4 hours**, potentially longer under memory pressure. Do not run it concurrently
with the full audit.

```powershell
& C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe -u -m scripts.test_uncertainty --acknowledge-long-analysis 2>&1 | Tee-Object -FilePath runs/final_studies/r0-final-v1-test-uncertainty.console.log
if ($LASTEXITCODE -ne 0) { throw 'Uncertainty analysis failed. Preserve partial output for review.' }
```

Completion requires `r0-final-v1-test/uncertainty-v1/complete.json` with successful
status and valid hashes, and `TEST UNCERTAINTY COMPLETE`. No automatic resume or
overwrite is supported. Preserve all bootstrap draws, per-replicate results,
seeds, uncertainty report, completion marker and console log.

```bash
tail -f runs/final_studies/r0-final-v1-test-uncertainty.console.log
```

## Remaining final report

Once the independent audit passes, populate the population, discrimination,
calibration, operational and evidence-stratum tables from the verified result.
Compare descriptively with the frozen validation reference, distinguish unique
episodes from overlapping positive windows, and disclose uncertainty and sparse
event limitations. Report warning time to alarm initiation retrospectively
classified Level 3, not ambulance dispatch or physiological crisis onset.

The paper-ready paragraph and the ten requested scientific conclusions remain
pending verification. Problem 10 is not yet declared scientifically complete.
No further model scoring is needed; only the read-only artifact audit and saved-
prediction uncertainty analysis remain before final interpretation.
