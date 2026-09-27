# Final R0 training verification and validation handoff

Final run: `runs/final_studies/r0-final-v1`.
**Final R0 training is valid and complete; the model is ready for canonical
validation.** No post-SSL stage is missing. No model/scaler was recomputed, no
fitting was performed during this review, and no validation or test outcomes were
accessed. The log ends with `Final R0 training complete: complete.json verified.`

## Independent checks

The read-only `scripts/audit_final_training.py` completed in approximately seven
seconds. Its private result is `runs/final_studies/r0-final-v1-audit.json`, outside
the immutable model directory. It verified all 17 completion-manifest hashes,
the frozen final specification/effective configuration, source-code hashes,
canonical task/run/TRAIN-export fingerprints, projection and feature order,
and training Git commit `2f3b8255a1c25057d981366cdfa9f20cb5e62442`.

Completion-marker SHA-256:
`19f5093d925e0beb39fc41114bbd32f473b8878ec311b7d2207a7410e50c0b3f`.
Final-specification semantic SHA-256:
`e297e15c2b5312ff55b7b3081b0743f009e067927d9745c4bd4a17629a6dbd63`.

The ten-way partition was independently reconstructed from canonical TRAIN
metadata. All SSL/scaler/head manifest patients belong to its fitting partition;
all checkpoint patients belong to its separate holdout. No canonical validation
or test patient participates in either partition. Every manifest reference was
checked for patient membership, sample index bounds, sample-ID format/consistent
mapping, and within-batch duplication. Head IDs/labels/weights agree with saved
embeddings and episode-linked training manifests; episode IDs belong to the
corresponding training patient.

The selected draw is the earliest minimum of the complete scheduled checkpoint
history. The saved risk encoder exactly equals the saved SSL encoder. Exported
head coefficients agree with the converged fit; all saved head scores reproduce
exactly, including repeated inference. Reloaded full-model inference agrees
exactly on a small synthetic input. The objective and standardized-coordinate
gradient were independently recalculated from saved training embeddings.

Scope limitation: this was a saved-artifact audit, not a second private-data pass.
Raw scaler statistics, checkpoint losses, physiological embedding extraction and
the corruption stream were not recomputed from patient shards. Their identities
are bound by the original source, manifests, seeds and artifact hashes. No claim
of replaying all private preprocessing or all 4,000 draws is made.

## Training summary

| Quantity | Result |
|---|---:|
| Canonical training patients | 8,704 |
| Fitting partition | 7,833 |
| Checkpoint holdout partition | 871 |
| Patients actually sampled for SSL | 6,837 |
| Unique SSL windows | 255,672 |
| SSL window exposures | 255,944 |
| SSL draws / contributing optimizer updates | 4,000 / 4,000 |
| Artificially masked target cells | 4,978,801 |
| Window exposures with no SSL targets | 62,345 (~24.36%) |
| Fixed checkpoint windows / sampled holdout patients | 1,024 / 61 |
| Scaler windows / patients | 4,096 / 251 |
| Unique scaler bucket rows | 365,838 |
| Head examples / patients | 8,287 / 7,833 |
| Positive head windows / unique linked episodes | 454 / 110 |

Only the 61 sampled holdout patients enter fixed checkpoint loss; all 871 remain
excluded from fitting. This is the frozen 16-batch checkpoint procedure, not a
deviation. SSL need not expose every fitting patient within the fixed draw budget.
Short patient pools explain exposures slightly below 4,000 × 64. Repeated windows
and masked cells are not independent clinical events.

Initial held reconstruction loss was **0.411633**. The selected checkpoint is
**draw 3,800**, held loss **0.199406**. Draw 4,000 held loss is **0.201267**; the
final training-batch loss is **0.277486**, which uses another batch/mask and should
not be interpreted as a comparable held estimate. The later selected checkpoint
is compatible with the prespecified extension; it does not establish an SSL gain
in Level-3 discrimination.

The logistic head took **60 iterations**, reducing the penalized training objective
from **0.0003501998380** to **0.0003425853144**. Independently recomputed maximum
absolute gradient is **1.59278e-10**, below the **1e-8** gate. No encoder parameters
changed during head fitting. All six scaler channels had observations.

Recorded runtime after preparation, including SSL/head export, was **1,306.80 s
(21.78 min)**. The provenance-to-completion file timestamp interval is approximately
**1,960.36 s (32.67 min)**, a whole-stage estimate rather than a separately measured
timer. CPU: four threads; no GPU fitting. Saved model-run artifacts occupy
**84,222,775 bytes (~80.32 MiB)**. Peak RAM was not recorded and is **unavailable**.
No protocol deviation requiring correction was found.

## Manual canonical validation handoff

This is the already frozen Problem 9 procedure; no model, calibration family,
threshold candidates, cooldown or metrics are changed. First score the entire
canonical validation stream. Then fit intercept calibration on the designated
validation-calibration patients and develop the threshold on the disjoint
validation-policy patients. The selected point must satisfy <=0.1 unmatched
alerts per supported patient-day. Matching remains first threshold crossing,
four-hour cooldown and one-to-one earliest unmatched episode in `(t,t+4h]`.

These are long/full-private-data operations and **were not executed**. Run from
PowerShell; failure in scoring prevents policy development from starting:

```powershell
Set-Location 'C:\Users\eldar\Projects\AI-CVD'
$python = 'C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe'
$scoreLog = 'runs/final-r0-v1-validation-scoring.log'
& $python -u -m src.final_study.cli score-validation --model-run runs/final_studies/r0-final-v1 --output runs/final_studies/r0-final-v1-validation-scores 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    Add-Content -LiteralPath $scoreLog -Value $line -Encoding utf8
}
if ($LASTEXITCODE -ne 0) { throw 'Validation scoring failed; preserve partial outputs and log.' }

$policyLog = 'runs/final-r0-v1-validation-policy.log'
& $python -u -m src.final_study.cli develop-validation --model-run runs/final_studies/r0-final-v1 --predictions runs/final_studies/r0-final-v1-validation-scores --output runs/final_studies/r0-final-v1-validation-policy 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    Add-Content -LiteralPath $policyLog -Value $line -Encoding utf8
}
if ($LASTEXITCODE -ne 0) { throw 'Validation policy development failed; preserve outputs and log.' }
```

Planning estimates remain **2–12 hours for scoring** and **10–60 minutes for
calibration/policy development**, with several GB RAM possible. Actual throughput
and stream size determine duration; validation outcomes were not inspected to
refine these estimates. Scoring logs each completed patient. Policy development
can remain quiet while evaluating its fixed threshold candidates.

Monitor in separate Git Bash windows:

```bash
tail -F /c/Users/eldar/Projects/AI-CVD/runs/final-r0-v1-validation-scoring.log
tail -F /c/Users/eldar/Projects/AI-CVD/runs/final-r0-v1-validation-policy.log
```

Completion requires successful exits and both stage markers:

- `runs/final_studies/r0-final-v1-validation-scores/complete.json`
- `runs/final_studies/r0-final-v1-validation-policy/complete.json`

The final console message is `Validation development complete; no test evaluation
performed.` Preserve both entire output directories and both logs. In particular,
retain prediction IDs/scores/labels/evidence flags, patient manifests, provenance,
`calibration.json`, `threshold.json`, `policy_development_metrics.json` and completion
hashes. The later report will include raw/calibrated AP/AUROC, calibration diagnostics,
unique-episode sensitivity, alert precision, unmatched alerts per supported day,
coverage, lead times and the predefined evidence strata.

**Resume behavior:** neither validation stage supports partial resume; existing
output directories are rejected. Do not delete or overwrite interrupted output.
Request recovery. If scoring completes but policy development has not started,
run only the second command block against the completed scoring directory.
Its hashes and model binding are verified before use.

Final R0 training and Problem 9 training are complete. The model is ready for
canonical validation; there is no blocking correction. Validation itself remains
unexecuted and unverified. Problem 10/test evaluation has not begun.
