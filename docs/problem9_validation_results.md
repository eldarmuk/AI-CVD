# Canonical validation verification — pending full audit

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
